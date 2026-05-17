# onnxruntime-bnlang

Bnlang binding for [ONNX Runtime](https://onnxruntime.ai).

> **Unofficial third-party binding.** Not affiliated with Microsoft.
> Consumes [official ORT prebuilt releases](https://github.com/microsoft/onnxruntime/releases).

*এই README-এর বাংলা সংস্করণ — [README.md](README.md)*

## Quick start

```bnl
import "onnxruntime-bnlang" as ort;

print(ort.version);
print(ort.providers());

var sess = ort.InferenceSession.open("model.onnx", { log_severity_level: 3 });

var info = sess.info();
var out  = sess.run({
    "input_ids":      { dtype: "int64", shape: [1, 16], data: [...] },
    "attention_mask": { dtype: "int64", shape: [1, 16], data: [...] }
});
// out["logits"] is { dtype, shape, handle } — materialize with tensor_to_data:
var logits = ort.tensor_to_data(out["logits"]["handle"]);
ort.tensor_close(out["logits"]["handle"]);

sess.close();
```

## Exports

| Name | Kind |
|---|---|
| `version` | string |
| `providers()` | function → list |
| `session_open(path, options)` | function → handle |
| `session_close(handle)` | function |
| `session_info(handle)` | function → map |
| `session_run(handle, feeds)` | function → outputs map |
| `tensor_to_data(handle)` | function → list |
| `tensor_close(handle)` | function |
| `InferenceSession.open(path, options)` | factory → session object |
| `sess.info() / sess.run(feeds) / sess.close()` | session-instance methods |

## Status (v1.0.0)

**Working:**
- Session lifecycle: open / info / run / close
- Tensor dtypes: float32, float64, float16, int64, int32, uint8, bool
- Session options: `log_severity_level`, `intra_op_num_threads`, `inter_op_num_threads`, `execution_providers`
- Tensor handle table (zero-copy KV-cache reuse for downstream LLM loops)

**Coming next:**
- More execution providers (CUDA, CoreML)
- Fetch-list filtering (pick which outputs to compute)
- Profiling + IO binding

## Build (local)

```powershell
# Windows
bnl script/install.bnl    # download ORT prebuilt into deps/windows-x64/
.\build.ps1                # cmake configure + build  ->  build/windows-x64/*.dll
```

```bash
# macOS / Linux
bnl script/install.bnl
./build.sh
```

## Cross-platform packaging

Built per platform; the bnl runtime picks the right artifact from `bnl.json`'s `targets` map at `import` time:

| Triple        | Build artifact                                |
|---------------|-----------------------------------------------|
| `windows-x64` | `build/windows-x64/onnxruntime-bnlang.dll`    |
| `linux-x64`   | `build/linux-x64/onnxruntime-bnlang.so`       |
| `darwin-arm64`| `build/darwin-arm64/onnxruntime-bnlang.dylib` |
| `darwin-x64`  | `build/darwin-x64/onnxruntime-bnlang.dylib`   |

`LOAD_WITH_ALTERED_SEARCH_PATH` in bnl core ensures the plugin's own directory is searched first for its dependencies, ahead of `System32` / `PATH`.

## Layout

```
bnl.json                 manifest (main + targets map)
CMakeLists.txt           build config
CMakePresets.json        one preset per platform

lib/
  index.bnl              public API (English + Bangla re-exports)
  session.bnl            InferenceSession + session-instance methods

src/                     C++ binding (excluded from published tarball)
  bnl/plugin.h           C ABI contract
  main.cpp               bnl_load entry
  session.{h,cpp}        Ort::Session handle table
  tensor.{h,cpp}         bnl_value <-> Ort::Value + handle table
  session_options.{h,cpp}

deps/<triple>/           ORT prebuilt (gitignored; install populates)
build/<triple>/          cmake output (gitignored)

script/
  install.bnl            download + verify + extract ORT prebuilt
  install-metadata.bnl   URLs + sha256 per platform
  build.bnl              thin wrapper around `cmake --preset`
```

## License

MIT. ORT prebuilts are MIT-licensed by Microsoft. See `NOTICES.md` for third-party attribution.
