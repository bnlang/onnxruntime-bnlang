# onnxruntime-bnlang

[ONNX Runtime](https://onnxruntime.ai)-এর জন্য Bnlang বাইন্ডিং।

> **এটি একটি বেসরকারি তৃতীয় পক্ষীয় বাইন্ডিং।** Microsoft-এর সাথে সম্পর্কিত নয়।
> [অফিসিয়াল ORT prebuilt রিলিজ](https://github.com/microsoft/onnxruntime/releases) ব্যবহার করে।

*Read this in English — [README.en.md](README.en.md)*

## দ্রুত শুরু

```bnl
import "onnxruntime-bnlang" as ort;

লিখুন(ort.সংস্করণ);
লিখুন(ort.প্রদানকারী());

ধরি সেশন = ort.অনুমান_সেশন.খুলুন("model.onnx", { log_severity_level: 3 });

ধরি তথ্য = সেশন.তথ্য();
ধরি আউট  = সেশন.চালান({
    "input_ids":      { dtype: "int64", shape: [1, 16], data: [...] },
    "attention_mask": { dtype: "int64", shape: [1, 16], data: [...] }
});
// আউট["logits"] হলো { dtype, shape, handle } — টেনসর_ডেটা দিয়ে materialize:
ধরি লজিটস = ort.টেনসর_ডেটা(আউট["logits"]["handle"]);
ort.টেনসর_বন্ধ_করুন(আউট["logits"]["handle"]);

সেশন.বন্ধ_করুন();
```

## Export-গুলো

| নাম | ধরন |
|---|---|
| `সংস্করণ` | string |
| `প্রদানকারী()` | function → list |
| `সেশন_খুলুন(path, options)` | function → handle |
| `সেশন_বন্ধ_করুন(handle)` | function |
| `সেশন_তথ্য(handle)` | function → map |
| `সেশন_চালান(handle, feeds)` | function → outputs map |
| `টেনসর_ডেটা(handle)` | function → list |
| `টেনসর_বন্ধ_করুন(handle)` | function |
| `অনুমান_সেশন.খুলুন(path, options)` | factory → session object |
| `সেশন.তথ্য() / সেশন.চালান(feeds) / সেশন.বন্ধ_করুন()` | session-instance method |

## অবস্থা (v1.0.0)

**যা কাজ করে:**
- Session lifecycle: খুলুন / তথ্য / চালান / বন্ধ_করুন
- Tensor dtype: float32, float64, float16, int64, int32, uint8, bool
- Session options: `log_severity_level`, `intra_op_num_threads`, `inter_op_num_threads`, `execution_providers`
- Tensor handle table (downstream LLM loop-এর জন্য zero-copy KV-cache reuse)

**পরবর্তী:**
- আরও execution provider (CUDA, CoreML)
- Fetch-list filtering (কোন output গুলো compute হবে সেটা বেছে নেওয়া)
- Profiling + IO binding

## লোকাল বিল্ড

```powershell
# Windows
bnl script/install.bnl    # deps/windows-x64/ এ ORT prebuilt নামায়
.\build.ps1                # cmake configure + build  ->  build/windows-x64/*.dll
```

```bash
# macOS / Linux
bnl script/install.bnl
./build.sh
```

## ক্রস-প্ল্যাটফর্ম

প্রতি platform-এ আলাদাভাবে build হয়; বাইন্ডিং রানটাইম import-এর সময় `bnl.json`-এর `targets` ম্যাপ থেকে সঠিক artifact বাছে। সমর্থিত triple:

| Triple        | বিল্ড artifact                                |
|---------------|-----------------------------------------------|
| `windows-x64` | `build/windows-x64/onnxruntime-bnlang.dll`    |
| `linux-x64`   | `build/linux-x64/onnxruntime-bnlang.so`       |
| `darwin-arm64`| `build/darwin-arm64/onnxruntime-bnlang.dylib` |
| `darwin-x64`  | `build/darwin-x64/onnxruntime-bnlang.dylib`   |

bnl core-এর `LOAD_WITH_ALTERED_SEARCH_PATH` plugin-এর নিজস্ব directory আগে খোঁজে, `System32` / `PATH`-এর আগে।

## লেআউট

```
bnl.json                 manifest (main + targets ম্যাপ)
CMakeLists.txt           build config
CMakePresets.json        প্রতি platform-এ একটি preset

lib/
  index.bnl              public API (ইংরেজি + বাংলা re-export)
  session.bnl            InferenceSession / অনুমান_সেশন

src/                     C++ source (publish-এ আসে না)
  bnl/plugin.h           C ABI
  main.cpp               bnl_load entry
  session.{h,cpp}        Ort::Session handle table
  tensor.{h,cpp}         bnl_value <-> Ort::Value + handle table
  session_options.{h,cpp}

deps/<triple>/           ORT prebuilt (gitignored; install দ্বারা populate হয়)
build/<triple>/          cmake output (gitignored)

script/
  install.bnl            ORT prebuilt fetch + verify + extract
  install-metadata.bnl   প্রতি platform-এ URL + sha256
  build.bnl              cmake-এর পাতলা wrapper
```

## লাইসেন্স

MIT. ORT prebuilt গুলো Microsoft-এর MIT লাইসেন্সে। তৃতীয় পক্ষীয় attribution `NOTICES.md`-এ আছে।
