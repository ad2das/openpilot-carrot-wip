// tlv_snpe_bench — minimal SNPE runner for the carrot TLV models.
// SNPE call sequence follows comma.ai openpilot 08445d018
// selfdrive/modeld/runners/snpemodel.cc (container open, SNPEBuilder,
// user-supplied float buffers), without the openpilot dependencies.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/DlEnums.hpp>
#include <DlSystem/IBufferAttributes.hpp>
#include <DlSystem/IUserBuffer.hpp>
#include <DlSystem/IUserBufferFactory.hpp>
#include <DlSystem/TensorShape.hpp>
#include <DlSystem/UserBufferMap.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>

namespace {

struct Tensor {
  std::string name;
  std::vector<size_t> dims;
  size_t count = 0;
  std::vector<float> data;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> buffer;
};

struct Result {
  std::string status = "fail";
  std::string error = "unknown_error";
  std::string snpe_version;
  std::string runtime;
  bool runtime_available = false;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  double load_s = 0.0;
  double first_ms = 0.0;
  double median_ms = 0.0;
  double p90_ms = 0.0;
  int runs = 0;
  int frames = 0;
  std::vector<double> frame_ms;
};

double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

std::string quote(const std::string &s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += char(c);
        }
    }
  }
  return out + "\"";
}

std::string dims_json(const std::vector<size_t> &dims) {
  std::string out = "[";
  for (size_t i = 0; i < dims.size(); i++) {
    if (i) out += ", ";
    out += std::to_string(dims[i]);
  }
  return out + "]";
}

std::string tensors_json(const std::vector<Tensor> &ts) {
  std::string out = "[";
  for (size_t i = 0; i < ts.size(); i++) {
    if (i) out += ", ";
    out += "{\"name\": " + quote(ts[i].name) + ", \"dims\": " + dims_json(ts[i].dims) + "}";
  }
  return out + "]";
}

std::string json(const Result &r) {
  char buf[256];
  std::string out = "{\n";
  out += "  \"status\": " + quote(r.status) + ",\n";
  out += "  \"error\": " + quote(r.error) + ",\n";
  out += "  \"snpe_version\": " + quote(r.snpe_version) + ",\n";
  out += "  \"runtime\": " + quote(r.runtime) + ",\n";
  out += std::string("  \"runtime_available\": ") + (r.runtime_available ? "true" : "false") + ",\n";
  out += "  \"inputs\": " + tensors_json(r.inputs) + ",\n";
  out += "  \"outputs\": " + tensors_json(r.outputs) + ",\n";
  snprintf(buf, sizeof(buf), "  \"load_s\": %.6f,\n  \"first_ms\": %.6f,\n  \"median_ms\": %.6f,\n  \"p90_ms\": %.6f,\n",
           r.load_s, r.first_ms, r.median_ms, r.p90_ms);
  out += buf;
  out += "  \"runs\": " + std::to_string(r.runs) + ",\n";
  out += "  \"frames\": " + std::to_string(r.frames) + ",\n";
  out += "  \"frame_ms\": [";
  for (size_t i = 0; i < r.frame_ms.size(); i++) {
    if (i) out += ", ";
    snprintf(buf, sizeof(buf), "%.3f", r.frame_ms[i]);
    out += buf;
  }
  out += "]\n}\n";
  return out;
}

bool write_atomic(const std::string &path, const std::string &body) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(body.data(), std::streamsize(body.size()));
    f.flush();
    if (!f) return false;
  }
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

std::vector<size_t> strides_for(const std::vector<size_t> &dims) {
  std::vector<size_t> strides(dims.size(), sizeof(float));
  for (size_t i = dims.size(); i > 1; i--) strides[i - 2] = strides[i - 1] * dims[i - 1];
  return strides;
}

size_t product(const std::vector<size_t> &dims) {
  size_t p = 1;
  for (size_t d : dims) p *= d;
  return p;
}

// Reads exactly `expect` float32 LE values from `path`. Returns 0 on success and
// sets `err` otherwise; a byte size other than `expect * 4` is `input_size_mismatch`
// and the file is never read.
int read_float_file(const std::string &path, size_t expect, std::vector<float> &out,
                    std::string &err, std::streamoff &got_bytes) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { err = "input_open"; return 1; }
  got_bytes = f.tellg();
  if (got_bytes < 0) { err = "input_open"; return 1; }
  const size_t want_bytes = expect * sizeof(float);
  if (size_t(got_bytes) != want_bytes) { err = "input_size_mismatch"; return 2; }
  f.seekg(0, std::ios::beg);
  out.resize(expect);
  f.read(reinterpret_cast<char *>(out.data()), std::streamsize(want_bytes));
  if (!f) { err = "input_read"; return 1; }
  return 0;
}

// Reads a --input_list file: one "<raw.f32> <dump.f32>" per non-empty line.
// Returns false with `err` set on malformed lines.
bool read_input_list(const std::string &path, std::vector<std::pair<std::string, std::string>> &out,
                     std::string &err) {
  std::ifstream f(path);
  if (!f) { err = "input_list_open"; return false; }
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    lineno++;
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
    if (line.empty()) continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size()) {
      err = "input_list_bad_line:" + std::to_string(lineno);
      return false;
    }
    out.emplace_back(line.substr(0, sp), line.substr(sp + 1));
  }
  return true;
}

struct XorShift32 {
  uint32_t s;
  explicit XorShift32(uint32_t seed) : s(seed) {}
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  // uniform in [0, 1)
  float unit() { return float(next() >> 8) * (1.0f / 16777216.0f); }
};

}  // namespace

int main(int argc, char **argv) {
  Result res;
  std::string dlc_path, out_path, dump_path, runtime_arg;
  std::string input_list_path;
  std::vector<std::pair<std::string, std::string>> input_files;
  int runs = 0;

  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    const bool has_next = (i + 1) < argc;
    if (a == "--dlc" && has_next) {
      dlc_path = argv[++i];
    } else if (a == "--runtime" && has_next) {
      runtime_arg = argv[++i];
    } else if (a == "--runs" && has_next) {
      runs = std::atoi(argv[++i]);
    } else if (a == "--out" && has_next) {
      out_path = argv[++i];
    } else if (a == "--dump" && has_next) {
      dump_path = argv[++i];
    } else if (a == "--input" && has_next) {
      const std::string spec = argv[++i];
      const size_t eq = spec.find('=');
      if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
        res.error = "bad_input_argument";
        res.runtime = runtime_arg;
        fprintf(stderr, "tlv_snpe_bench: status=fail error=bad_input_argument\n");
        return 64;
      }
      input_files.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
    } else if (a == "--input_list" && has_next) {
      input_list_path = argv[++i];
    } else {
      fprintf(stderr, "usage: tlv_snpe_bench --dlc <path> --runtime cpu|dsp --runs <N> --out <result.json> [--dump <out.f32>] [--input <tensor>=<raw.f32>]... [--input_list <file>]\n");
      return 64;
    }
  }
  res.runtime = runtime_arg;

  auto finish = [&](int code) {
    if (!out_path.empty()) {
      if (!write_atomic(out_path, json(res))) {
        fprintf(stderr, "tlv_snpe_bench: failed to write %s\n", out_path.c_str());
        if (code == 0) code = 1;
      }
    }
    fprintf(stderr, "tlv_snpe_bench: status=%s error=%s\n", res.status.c_str(), res.error.c_str());
    return code;
  };

  if (dlc_path.empty() || out_path.empty() || (runtime_arg != "cpu" && runtime_arg != "dsp")) {
    res.error = "bad_arguments";
    return finish(64);
  }

  std::vector<std::pair<std::string, std::string>> frames;
  if (!input_list_path.empty()) {
    if (!input_files.empty() || !dump_path.empty()) {
      res.error = "input_list_conflicts_with_input_or_dump";
      fprintf(stderr, "tlv_snpe_bench: usage: --input_list cannot be combined with --input or --dump\n");
      return finish(64);
    }
    std::string list_err;
    if (!read_input_list(input_list_path, frames, list_err) || frames.empty()) {
      res.error = list_err.empty() ? "input_list_empty" : list_err;
      return finish(1);
    }
  } else if (runs < 1) {
    res.error = "bad_arguments";
    return finish(64);
  }

  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
  if (runtime_arg == "dsp") runtime = zdl::DlSystem::Runtime_t::DSP;

  res.snpe_version = zdl::SNPE::SNPEFactory::getLibraryVersion().asString().c_str();
  res.runtime_available = zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime);
  if (!res.runtime_available) {
    res.error = "runtime_unavailable";
    return finish(2);
  }

  const double t_load0 = now_s();

  std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(dlc_path);
  if (!container) {
    res.error = std::string("container_open: ") + zdl::DlSystem::getLastErrorString();
    return finish(1);
  }

  std::unique_ptr<zdl::SNPE::SNPE> snpe;
  {
    zdl::SNPE::SNPEBuilder builder(container.get());
    snpe = builder.setOutputLayers({})
                  .setRuntimeProcessor(runtime)
                  .setUseUserSuppliedBuffers(true)
                  .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                  .build();
  }
  if (!snpe) {
    res.error = std::string("build: ") + zdl::DlSystem::getLastErrorString();
    return finish(1);
  }

  const auto &in_names_opt = snpe->getInputTensorNames();
  if (!in_names_opt) {
    res.error = "no_input_names";
    return finish(1);
  }
  const auto &out_names_opt = snpe->getOutputTensorNames();
  if (!out_names_opt) {
    res.error = "no_output_names";
    return finish(1);
  }
  const zdl::DlSystem::StringList &in_names = *in_names_opt;
  const zdl::DlSystem::StringList &out_names = *out_names_opt;

  if (!input_list_path.empty() && in_names.size() != 1) {
    res.error = "input_list_needs_one_input";
    return finish(1);
  }

  zdl::DlSystem::UserBufferEncodingFloat float_encoding;
  zdl::DlSystem::IUserBufferFactory &ub_factory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  zdl::DlSystem::UserBufferMap in_map, out_map;

  for (size_t i = 0; i < in_names.size(); i++) {
    Tensor t;
    t.name = in_names.at(i);
    const auto &dims_opt = snpe->getInputDimensions(t.name.c_str());
    if (!dims_opt) {
      res.error = "no_input_dims: " + t.name;
      return finish(1);
    }
    const zdl::DlSystem::TensorShape &shape = *dims_opt;
    for (size_t d = 0; d < shape.rank(); d++) t.dims.push_back(shape[d]);
    t.count = product(t.dims);
    t.data.assign(t.count, 0.0f);
    const std::vector<size_t> strides = strides_for(t.dims);
    t.buffer = ub_factory.createUserBuffer(t.data.data(), t.count * sizeof(float),
                                           zdl::DlSystem::TensorShape(strides), &float_encoding);
    if (!t.buffer) {
      res.error = std::string("create_input_buffer: ") + t.name + ": " + zdl::DlSystem::getLastErrorString();
      return finish(1);
    }
    in_map.add(t.name.c_str(), t.buffer.get());
    res.inputs.push_back(std::move(t));
  }

  for (size_t i = 0; i < out_names.size(); i++) {
    Tensor t;
    t.name = out_names.at(i);
    const auto attrs_opt = snpe->getInputOutputBufferAttributes(t.name.c_str());
    if (!attrs_opt) {
      res.error = "no_output_attributes: " + t.name;
      return finish(1);
    }
    const zdl::DlSystem::TensorShape &shape = (*attrs_opt)->getDims();
    for (size_t d = 0; d < shape.rank(); d++) t.dims.push_back(shape[d]);
    t.count = product(t.dims);
    t.data.assign(t.count, 0.0f);
    const std::vector<size_t> strides = strides_for(t.dims);
    t.buffer = ub_factory.createUserBuffer(t.data.data(), t.count * sizeof(float),
                                           zdl::DlSystem::TensorShape(strides), &float_encoding);
    if (!t.buffer) {
      res.error = std::string("create_output_buffer: ") + t.name + ": " + zdl::DlSystem::getLastErrorString();
      return finish(1);
    }
    out_map.add(t.name.c_str(), t.buffer.get());
    res.outputs.push_back(std::move(t));
  }

  res.load_s = now_s() - t_load0;

  // fixed input pattern, identical for every run; --input overrides individual
  // tensors with raw little-endian float32 files.
  {
    XorShift32 rng(7);
    for (Tensor &t : res.inputs) {
      for (size_t k = 0; k < t.count; k++) t.data[k] = rng.unit();
    }
    for (const auto &spec : input_files) {
      if (std::none_of(res.inputs.begin(), res.inputs.end(),
                       [&](const Tensor &t) { return t.name == spec.first; })) {
        res.error = "unknown_input:" + spec.first;
        return finish(1);
      }
    }
    for (const auto &spec : input_files) {
      for (Tensor &t : res.inputs) {
        if (t.name != spec.first) continue;
        std::vector<float> raw;
        std::string err;
        std::streamoff got = 0;
        const int rc = read_float_file(spec.second, t.count, raw, err, got);
        if (rc != 0) {
          if (err == "input_size_mismatch") {
            res.error = err + ":" + t.name;
            fprintf(stderr, "tlv_snpe_bench: %s expected_bytes=%zu got_bytes=%lld path=%s\n",
                    res.error.c_str(), t.count * sizeof(float), (long long)got, spec.second.c_str());
          } else {
            res.error = err + ":" + t.name;
            fprintf(stderr, "tlv_snpe_bench: %s path=%s\n", res.error.c_str(), spec.second.c_str());
          }
          return finish(1);
        }
        t.data = std::move(raw);
      }
    }
    for (Tensor &t : res.inputs) t.buffer->setBufferAddress(t.data.data());
  }

  if (!frames.empty()) {
    // One load, one execute per line: the raw feeds the single input and every
    // output is written to that line's dump. --runs is ignored (runs = N).
    Tensor &in_t = res.inputs.front();
    std::vector<double> frame_ms;
    frame_ms.reserve(frames.size());
    for (size_t k = 0; k < frames.size(); k++) {
      std::vector<float> raw;
      std::string err;
      std::streamoff got = 0;
      const int rc = read_float_file(frames[k].first, in_t.count, raw, err, got);
      if (rc != 0) {
        res.error = err + ":" + in_t.name + "@" + std::to_string(k);
        fprintf(stderr, "tlv_snpe_bench: %s path=%s\n", res.error.c_str(), frames[k].first.c_str());
        return finish(1);
      }
      in_t.data = std::move(raw);
      in_t.buffer->setBufferAddress(in_t.data.data());

      const double t0 = now_s();
      if (!snpe->execute(in_map, out_map)) {
        res.error = std::string("execute: ") + zdl::DlSystem::getLastErrorString();
        return finish(1);
      }
      frame_ms.push_back((now_s() - t0) * 1e3);

      std::ofstream f(frames[k].second, std::ios::binary | std::ios::trunc);
      if (!f) {
        res.error = "dump_open:" + frames[k].second;
        return finish(1);
      }
      for (const Tensor &t : res.outputs) {
        f.write(reinterpret_cast<const char *>(t.data.data()), std::streamsize(t.count * sizeof(float)));
      }
      f.flush();
      if (!f) {
        res.error = "dump_write:" + frames[k].second;
        return finish(1);
      }
    }
    res.frames = int(frames.size());
    res.frame_ms = frame_ms;
    res.runs = int(frames.size());
    res.first_ms = frame_ms[0];
    if (frame_ms.size() > 1) {
      std::vector<double> rest(frame_ms.begin() + 1, frame_ms.end());
      std::sort(rest.begin(), rest.end());
      const size_t n = rest.size();
      res.median_ms = (n % 2) ? rest[n / 2] : 0.5 * (rest[n / 2 - 1] + rest[n / 2]);
      size_t idx = size_t(std::ceil(0.9 * double(n))) - 1;
      res.p90_ms = rest[std::min(idx, n - 1)];
    } else {
      res.median_ms = res.first_ms;
      res.p90_ms = res.first_ms;
    }
    res.status = "pass";
    res.error = "";
    return finish(0);
  }

  std::vector<double> elapsed_ms;
  for (int run = 0; run < runs; run++) {
    const double t0 = now_s();
    if (!snpe->execute(in_map, out_map)) {
      res.error = std::string("execute: ") + zdl::DlSystem::getLastErrorString();
      res.runs = run;
      return finish(1);
    }
    elapsed_ms.push_back((now_s() - t0) * 1e3);
  }
  res.runs = runs;
  res.first_ms = elapsed_ms[0];

  if (runs > 1) {
    std::vector<double> rest(elapsed_ms.begin() + 1, elapsed_ms.end());
    std::sort(rest.begin(), rest.end());
    const size_t n = rest.size();
    res.median_ms = (n % 2) ? rest[n / 2] : 0.5 * (rest[n / 2 - 1] + rest[n / 2]);
    size_t idx = size_t(std::ceil(0.9 * double(n))) - 1;
    res.p90_ms = rest[std::min(idx, n - 1)];
  } else {
    res.median_ms = res.first_ms;
    res.p90_ms = res.first_ms;
  }

  if (!dump_path.empty()) {
    std::ofstream f(dump_path, std::ios::binary | std::ios::trunc);
    if (!f) {
      res.error = "dump_open: " + dump_path;
      return finish(1);
    }
    for (const Tensor &t : res.outputs) {
      f.write(reinterpret_cast<const char *>(t.data.data()), std::streamsize(t.count * sizeof(float)));
    }
    f.flush();
    if (!f) {
      res.error = "dump_write: " + dump_path;
      return finish(1);
    }
  }

  res.status = "pass";
  res.error = "";
  return finish(0);
}
