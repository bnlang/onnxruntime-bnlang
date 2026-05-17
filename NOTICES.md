# Third-party notices — onnxruntime-bnlang

This package's binary plugin dynamically links to and consumes prebuilt
binaries from the projects below. They are downloaded at install time by
`script/install.bnl` from their official distribution channels (NuGet,
GitHub Releases) and remain governed by their original licenses.

---

## ONNX Runtime

- **Project:** Microsoft ONNX Runtime
- **Upstream:** https://github.com/microsoft/onnxruntime
- **License:** MIT
- **Copyright:** Copyright (c) Microsoft Corporation

Used as the inference engine. We download Microsoft's prebuilt
`onnxruntime.dll` / `onnxruntime_providers_shared.dll` at install time and
dynamically link to it from our plugin. We do not redistribute the ORT
binaries; users fetch them from Microsoft's official packages.

The full MIT license text is available at
https://github.com/microsoft/onnxruntime/blob/main/LICENSE

---

## DirectML

- **Project:** DirectML
- **Upstream:** part of Microsoft Windows (since Windows 10 1903)
- **License:** Microsoft Software License Terms (Windows component)

`DirectML.dll` is shipped with Windows itself and is not redistributed by
this package. It is loaded from the system when `execution_providers`
includes `"DML"`.
