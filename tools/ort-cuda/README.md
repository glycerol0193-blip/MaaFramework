# Experimental ORT CUDA OCR configuration

`WITH_ORT_CUDA` defaults to `OFF`. This option selects the ONNX Runtime CUDA EP for OCR while retaining CPU image preprocessing and postprocessing. It does not enable FastDeploy's native CUDA preprocessing or custom CUDA kernels.

Use a matching CUDA-enabled ONNX Runtime, CUDA/cuDNN DLLs, and the existing ORT-backed FastDeploy build. Keep the dependency versions and DLL import names compatible. Place experiment dependencies in a separate application directory; do not replace system or another MAA installation's DLLs.

Only the pinned standard-ONNX OCR models were validated. Models requiring FastDeploy custom CUDA operators need a separately validated GPU build. Inspect model operators before enabling this experiment for another resource pack.

Configure with `-DWITH_ORT_CUDA=ON` and build `Debug`. The default build and CPU selection remain unchanged. GPU placement must be verified with a profiler; successful configuration alone is not proof of GPU execution.

## 简体中文

此实验选项默认关闭，只配置 ORT CUDA 推理，图像前后处理仍在 CPU。它不代表启用了 FastDeploy 自己的 CUDA 前处理或自定义 CUDA 算子。

请隔离部署版本与导入名称匹配的 ORT GPU、CUDA/cuDNN 及 FastDeploy 运行库。不要覆盖系统或其他 MAA 安装中的 DLL。当前验证仅覆盖固定版本的标准 ONNX OCR 模型；其他资源包须先检查算子与结果一致性。配置成功不等于 GPU 运算已生效，需用 profiler 核实。

## Existing resource options

Select the existing `MaaInferenceExecutionProvider_CUDA` value and a numeric CUDA device ordinal before loading resources. No public API or Pipeline field is added.

```cpp
MaaInferenceExecutionProvider ep = MaaInferenceExecutionProvider_CUDA;
MaaInferenceDevice device = MaaInferenceDevice_0;
MaaResourceSetOption(resource, MaaResOption_InferenceExecutionProvider, &ep, sizeof(ep));
MaaResourceSetOption(resource, MaaResOption_InferenceDevice, &device, sizeof(device));
```

The experiment configures `RuntimeOption.device` and `device_id` after explicitly selecting the ORT backend. It avoids the native-kernel build check in FastDeploy's `UseCuda` convenience setter; it does not add a CUDA backend implementation. Existing resource loading, failure fallback and session lifetime behavior are preserved.

加载资源前使用既有 CUDA EP 枚举和数字设备序号即可；没有增加公开 API 或 Pipeline 字段。此配置保留原有加载、回退与生命周期行为，不修改后端实现。
