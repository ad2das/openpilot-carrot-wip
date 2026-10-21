# snpe_tlv — Qualcomm SNPE runtime bundle + benchmark runner

Self-contained SNPE aarch64 runtime needed to run quantized `.dlc` models on the
comma three (QCOM2 / SD845) Hexagon DSP, plus `tlv_snpe_bench`, a small C++
runner that loads a DLC, runs it N times and writes a JSON report.

## Provenance

- Upstream commit: **carrot-openpilot `08445d018`** (comma.ai openpilot era, 2022)
- Original paths copied verbatim (`git cat-file`, byte-exact — see `SHA256SUMS`):
  - `third_party/snpe/aarch64-linux-gcc4.9/` → `larch64/`
  - `third_party/snpe/dsp/` → `dsp/`
  - `third_party/snpe/include/` → `include/`
  - `models/dmonitoring_model_q.dlc` → `models/dmonitoring_model_q.dlc`
- In upstream, `third_party/snpe/larch64` is a symlink to `aarch64-linux-gcc4.9`;
  here it is a real directory so the bundle needs no symlink.
- `SHA256SUMS` lists the sha256 of every copied file; each was checked against
  `git cat-file -p 08445d018:<original path> | sha256sum`.

## Contents

| path | files | bytes | what |
| --- | --- | --- | --- |
| `larch64/` | 8 | 11,452,270 | SNPE host runtime (aarch64-linux-gcc4.9) |
| `dsp/` | 6 | 5,883,380 | Hexagon DSP skel libraries (`libsnpe_dsp_skel.so`, `libsnpe_dsp_v65/v66_domains_v2_skel.so`, calculator skels) |
| `include/` | 41 | 188,231 | SNPE C++ headers (`DlContainer/`, `DlSystem/`, `SNPE/`, `SnpeUdo/`, `DiagLog/`, `PlatformValidator/`) |
| `models/` | 1 | 1,145,921 | comma's quantized driver-monitoring model, used as the test model |
| `src/tlv_snpe_bench.cc` | 1 | - | the runner |

`larch64/`:

```
20828 <-> libcalculator_domains.so
20828  libcalculator.so
20192 libsnpe_dsp_domains_v2.so
19392 libsnpe_adsp.so
1114072 libPlatformValidatorShared.so
9447816 libSNPE.so
123705 libatomic.so.1
648120 libsymphony-cpu.so
```

`dsp/`:

```
20828 libcalculator_domains_skel.so
20816 libcalculator_skel.so
1434132 libsnpe_dsp_domains_skel.so
1431408 libsnpe_dsp_skel.so
1492098 libsnpe_dsp_v65_domains_v2_skel.so
1492098 libsnpe_dsp_v66_domains_v2_skel.so
```

(Exact per-file byte counts and hashes: `SHA256SUMS`.)

`models/dmonitoring_model_q.dlc` is 1,145,921 B, inputs `input_img` `[1,6,320,160]`
float and `calib` `[1,3]` float, output `outputs` `[1,45]` float.

## Runtime environment on the device

As comma did on QCOM2 (`selfdrive/modeld/dmonitoringmodeld` at `08445d018`):

```
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:<bundle>/larch64
ADSP_LIBRARY_PATH=<bundle>/dsp/          # DSP skels, trailing slash required
```

The DSP runtime also needs the Hexagon `fastrpc` userspace + `cdsprpc`/`adsprpc`
device nodes; when they are missing SNPE reports the DSP runtime unavailable and
the runner exits 2 with `runtime_unavailable` (this is the expected PC behaviour).

## Build

Cross toolchain: `apt-get install g++-aarch64-linux-gnu libc6-arm64-cross`.

```
aarch64-linux-gnu-g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I third_party/snpe_tlv/include \
  third_party/snpe_tlv/src/tlv_snpe_bench.cc \
  -L third_party/snpe_tlv/larch64 -lSNPE \
  -Wl,-rpath,'$ORIGIN/../larch64' \
  -o third_party/snpe_tlv/bin/tlv_snpe_bench
```

`-D_GLIBCXX_USE_CXX11_ABI=0` is **required**: these 2022 SNPE libraries export the
old libstdc++ ABI (`IDlContainer::open(std::string const&)` mangles as
`_ZN3zdl11DlContainer12IDlContainer4openERKSs`, not `...B5cxx11...`). Without the
flag the link fails on every `zdl::...` API taking `std::string`. libstdc++ (not
libc++) is mandatory — `std::string` crosses the SNPE C++ ABI; `libSNPE.so` itself
needs only `GLIBC_2.17` and `GLIBCXX_3.4` … `GLIBCXX_3.4.20`.

`bin/` is git-ignored; the device build produces the binary itself.

## Runner

```
tlv_snpe_bench --dlc <path> --runtime cpu|dsp --runs <N> --out <result.json> [--dump <out.f32>]
```

- Loads the DLC via `IDlContainer::open`, builds with `SNPEBuilder`
  (`setOutputLayers({})`, `setRuntimeProcessor`, `setUseUserSuppliedBuffers(true)`,
  `setPerformanceProfile(HIGH_PERFORMANCE)`) — the same call sequence as
  openpilot's `snpemodel.cc`, minus the openpilot dependencies.
- Every input tensor (all of them, not just the first) is a float user buffer fed
  from a fixed xorshift32 generator seeded 7, values uniform in `[0, 1)`. Every
  output is a float user buffer, sized from `getInputOutputBufferAttributes`.
- `--dump` writes all outputs concatenated as little-endian float32, in
  output-name order.
- `--out` is written atomically (`.tmp` + rename) with keys `status`, `error`,
  `snpe_version`, `runtime`, `runtime_available`, `inputs`, `outputs`, `load_s`,
  `first_ms`, `median_ms`, `p90_ms`, `runs`.
- Exit codes: `0` pass, `2` requested runtime unavailable (`error: runtime_unavailable`),
  `1` SNPE error (`error` carries `zdl::DlSystem::getLastErrorString()`), `64` bad arguments.
- `build()` is attempted exactly once; a failure is reported, never retried.

## Licence

Qualcomm SNPE runtime redistributed as in comma.ai openpilot (`third_party/snpe`).
