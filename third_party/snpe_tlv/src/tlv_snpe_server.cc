// tlv_snpe_server — persistent SNPE runner for the carrot TLV detector.
//
// Derived from tlv_snpe_bench.cc (same SNPE call sequence: comma.ai openpilot
// 08445d018 selfdrive/modeld/runners/snpemodel.cc - container open, SNPEBuilder,
// user-supplied float buffers). Instead of loading a DLC per run, the model is
// built once and the process then serves frames over stdin/stdout, exchanging
// pixel input and head output through two mmap'ed files created by the parent:
//
//   tlvd -> server : 614400 B, uint8 NCHW RGB (1,3,320,640)
//   server -> tlvd : 1092000 B, float32 output tensors in output-name order
//
// Protocol (one line each, stdout, flushed):
//   READY <in_bytes> <out_bytes>
//   OK <seq> <exec_us>
//   ERR <seq> <message>
//   ERR init <message>            (then exit 2)
//
// No other output on stdout; diagnostics go to stderr.

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

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

// The parent's contract with this process (see the header comment).
constexpr size_t kInBytes = 320 * 640 * 3;
constexpr size_t kOutBytes = 273000 * sizeof(float);
constexpr size_t kOutFloats = kOutBytes / sizeof(float);
constexpr size_t kInChannels = 3;

struct Tensor {
  std::string name;
  std::vector<size_t> dims;
  size_t count = 0;
  std::vector<float> data;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> buffer;
};

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

// ERR init + exit 2. Everything that can go wrong before READY goes through
// here so the parent sees exactly one reason on the first line.
int die(const std::string &msg) {
  printf("ERR init %s\n", msg.c_str());
  fflush(stdout);
  fprintf(stderr, "tlv_snpe_server: init failed: %s\n", msg.c_str());
  return 2;
}

struct Mapped {
  void *addr = MAP_FAILED;
  int fd = -1;
  size_t size = 0;

  bool open_file(const std::string &path, size_t want, int prot) {
    const int flags = (prot & PROT_WRITE) ? O_RDWR : O_RDONLY;
    fd = ::open(path.c_str(), flags);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || size_t(st.st_size) != want) return false;
    addr = mmap(nullptr, want, prot, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) return false;
    size = want;
    return true;
  }
};

// NCHW uint8 -> NHWC float, exactly the conversion the probe validated against
// onnxruntime on the same pixels (REPORT-WSNPE11 T-D).
void nchw_u8_to_nhwc_f32(const uint8_t *src, float *dst, size_t height, size_t width) {
  const size_t plane = height * width;
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      const size_t nchw = y * width + x;
      float *out = dst + nchw * kInChannels;
      for (size_t c = 0; c < kInChannels; c++) out[c] = float(src[c * plane + nchw]);
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string dlc_path, runtime_arg, in_path, out_path;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    const bool has_next = (i + 1) < argc;
    if (a == "--dlc" && has_next) {
      dlc_path = argv[++i];
    } else if (a == "--runtime" && has_next) {
      runtime_arg = argv[++i];
    } else if (a == "--in" && has_next) {
      in_path = argv[++i];
    } else if (a == "--out" && has_next) {
      out_path = argv[++i];
    } else {
      fprintf(stderr, "usage: tlv_snpe_server --dlc <path> --runtime cpu|dsp --in <shm_in> --out <shm_out>\n");
      return 64;
    }
  }
  if (dlc_path.empty() || in_path.empty() || out_path.empty() ||
      (runtime_arg != "cpu" && runtime_arg != "dsp")) {
    fprintf(stderr, "tlv_snpe_server: bad arguments\n");
    return 64;
  }

  // Orphan protection: a pull-away SIGKILLs tlvd, and this process must not
  // outlive it on the device.
  prctl(PR_SET_PDEATHSIG, SIGKILL);

  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
  if (runtime_arg == "dsp") runtime = zdl::DlSystem::Runtime_t::DSP;
  if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) return die("runtime_unavailable");

  Mapped in_map, out_map;
  if (!in_map.open_file(in_path, kInBytes, PROT_READ)) return die("input_shm");
  if (!out_map.open_file(out_path, kOutBytes, PROT_READ | PROT_WRITE)) return die("output_shm");

  std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(dlc_path);
  if (!container) return die(std::string("container_open: ") + zdl::DlSystem::getLastErrorString());

  std::unique_ptr<zdl::SNPE::SNPE> snpe;
  {
    zdl::SNPE::SNPEBuilder builder(container.get());
    snpe = builder.setOutputLayers({})
                  .setRuntimeProcessor(runtime)
                  .setUseUserSuppliedBuffers(true)
                  .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                  .build();
  }
  if (!snpe) return die(std::string("build: ") + zdl::DlSystem::getLastErrorString());

  const auto &in_names_opt = snpe->getInputTensorNames();
  if (!in_names_opt) return die("no_input_names");
  const auto &out_names_opt = snpe->getOutputTensorNames();
  if (!out_names_opt) return die("no_output_names");
  const zdl::DlSystem::StringList &in_names = *in_names_opt;
  const zdl::DlSystem::StringList &out_names = *out_names_opt;
  if (in_names.size() != 1) return die("input_count");

  zdl::DlSystem::UserBufferEncodingFloat float_encoding;
  zdl::DlSystem::IUserBufferFactory &ub_factory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  zdl::DlSystem::UserBufferMap in_ub_map, out_ub_map;

  Tensor input;
  input.name = in_names.at(0);
  {
    const auto &dims_opt = snpe->getInputDimensions(input.name.c_str());
    if (!dims_opt) return die("no_input_dims: " + input.name);
    const zdl::DlSystem::TensorShape &shape = *dims_opt;
    for (size_t d = 0; d < shape.rank(); d++) input.dims.push_back(shape[d]);
    input.count = product(input.dims);
    if (input.count != kInBytes) return die("input_size");
    input.data.assign(input.count, 0.0f);
    const std::vector<size_t> strides = strides_for(input.dims);
    input.buffer = ub_factory.createUserBuffer(input.data.data(), input.count * sizeof(float),
                                               zdl::DlSystem::TensorShape(strides), &float_encoding);
    if (!input.buffer) return die(std::string("create_input_buffer: ") + zdl::DlSystem::getLastErrorString());
    in_ub_map.add(input.name.c_str(), input.buffer.get());
  }

  // H (dims[1]), W (dims[2]), C (dims[3]) of the NHWC input; the source pixels
  // are the NCHW planes of the same frame.
  if (input.dims.size() != 4 || input.dims[3] != kInChannels || input.dims[0] != 1) return die("input_layout");
  const size_t in_h = input.dims[1];
  const size_t in_w = input.dims[2];

  std::vector<Tensor> outputs;
  size_t out_total = 0;
  for (size_t i = 0; i < out_names.size(); i++) {
    Tensor t;
    t.name = out_names.at(i);
    const auto &attrs_opt = snpe->getInputOutputBufferAttributes(t.name.c_str());
    if (!attrs_opt) return die("no_output_attributes: " + t.name);
    const zdl::DlSystem::TensorShape &shape = (*attrs_opt)->getDims();
    for (size_t d = 0; d < shape.rank(); d++) t.dims.push_back(shape[d]);
    t.count = product(t.dims);
    out_total += t.count;
    t.data.assign(t.count, 0.0f);
    const std::vector<size_t> strides = strides_for(t.dims);
    t.buffer = ub_factory.createUserBuffer(t.data.data(), t.count * sizeof(float),
                                           zdl::DlSystem::TensorShape(strides), &float_encoding);
    if (!t.buffer) return die(std::string("create_output_buffer: ") + t.name + ": " + zdl::DlSystem::getLastErrorString());
    out_ub_map.add(t.name.c_str(), t.buffer.get());
    outputs.push_back(std::move(t));
  }
  if (out_total != kOutFloats) return die("output_size");

  printf("READY %zu %zu\n", kInBytes, kOutBytes);
  fflush(stdout);

  const uint8_t *src = static_cast<const uint8_t *>(in_map.addr);
  float *dst = static_cast<float *>(out_map.addr);
  char line[256];
  while (fgets(line, sizeof(line), stdin) != nullptr) {
    if (strncmp(line, "RUN ", 4) != 0) {
      fprintf(stderr, "tlv_snpe_server: ignoring line: %s", line);
      continue;
    }
    const unsigned long long seq = strtoull(line + 4, nullptr, 10);

    nchw_u8_to_nhwc_f32(src, input.data.data(), in_h, in_w);

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = snpe->execute(in_ub_map, out_ub_map);
    const auto t1 = std::chrono::steady_clock::now();
    const long long exec_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    if (!ok) {
      const std::string err = zdl::DlSystem::getLastErrorString();
      printf("ERR %llu %s\n", seq, err.empty() ? "execute" : err.c_str());
      fflush(stdout);
      continue;
    }

    size_t off = 0;
    for (const Tensor &t : outputs) {
      memcpy(dst + off, t.data.data(), t.count * sizeof(float));
      off += t.count;
    }
    printf("OK %llu %lld\n", seq, exec_us);
    fflush(stdout);
  }
  return 0;
}
