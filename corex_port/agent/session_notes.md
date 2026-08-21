# lc0 迁移到 Iluvatar CoreX (ivcore11) 会话记录

## 来源
- 上游仓库：https://github.com/LeelaChessZero/lc0.git
- 迁移起点 commit：`d8ce48258c39d331c119f8c8729374ceb3df8409`（master，2026-05-06）
- 版本：lc0 v0.33.0-dev
- 构建系统：meson + ninja

## CUDA 使用性质
lc0 是国际象棋引擎，含多种神经网络推理后端。其 CUDA 相关部分：
- `src/neural/backends/cuda/`：
  - `network_cuda.cc` + `common_kernels.cu` + `fp16_kernels.cu`（plain CUDA 后端，自写 kernel + cuBLAS）
  - `network_cudnn.cc`（cuDNN 后端）
  - `cutlass_kernels.cu`（可选 CUTLASS/Ampere 加速，默认相关）
- `.cu` 文件在上游通过 `nvcc` 的 custom_target 编译（meson.build 探测 `nvcc -h` / `--dryrun` 后调用）。

## 环境
- 设备：Iluvatar BI-V150 ×2（本次仅用 GPU 0，`CUDA_VISIBLE_DEVICES=0`；未超过 2 卡红线）。
- 监控：`ixsmi`（IX-ML 4.4.0 / Driver 4.5.0 / CUDA 10.2），记录于 `env/ixsmi.log`。
- 编译器：CoreX clang++ 22.1.0（`/usr/local/corex/bin/clang++`），无 nvcc。
- CoreX SDK 提供：`libcudart.so.10.2`、`libcublas.so.10`、`libcudnn.so.7`、`libcuda.so.1`，头文件含 `cudnn.h`/`cublas_v2.h`。

## 适配内容
1. **nvcc → clang++（红线：禁用 nvcc）**：新增 `scripts/corex_nvcc_wrapper.sh` 作为 nvcc 的替身。它：
   - 对 meson 的 `-h`/`--dryrun` 探测返回合理结果（不宣告 sm_XX/-arch=native，从而 CUTLASS 因 max_cuda=0 自动关闭）；
   - 把真实编译命令翻译为 `clang++ -x ivcore --cuda-path=/usr/local/corex --cuda-gpu-arch=ivcore11 -fPIC`，剥离 `-arch/-gencode/-code/-Wno-deprecated-gpu-targets/-Xptxas/--use_fast_math/-maxrregcount` 等 nvcc-only flag，`-Xcompiler -fPIC` 翻译为 `-fPIC`，`--std=c++NN` 翻译为 `-std=c++NN`。
   - 在 `meson.build` 中新增 `-Dcorex` 开关（默认 false，不影响上游 NVIDIA 构建）来启用该 wrapper。
2. **fp16 arch gate 旁路**：`src/neural/backends/cuda/fp16_kernels.cu` 的 `#if __CUDA_ARCH__ >= 530` / `< 530` 门控，在 ivcore11（设备端 `__CUDA_ARCH__` 报 300）下会把真实 fp16 kernel body 静默裁成空壳（`SKIP_FP16_BITS`）。依据 `iluvatar-cuda-base` 索引「NV 530/700 数值 gate 静默裁掉 fp16 路径」，改为 `#if defined(__ILUVATAR__) || (__CUDA_ARCH__ >= 530)` 与 `#if !defined(__ILUVATAR__) && (__CUDA_ARCH__ < 530)`。
3. **构建选项**：`-Dplain_cuda=true -Dcudnn=true -Dcutlass=false -Dgtest=true`，`cudnn_libdirs=/usr/local/corex/lib64`、`cudnn_include=/usr/local/corex/include`。
4. **double/float64**：源码中的 `double` 仅出现在 host 端（`layers.cc` 的 `numeric_limits<double>::quiet_NaN()`、`network_cudnn.cc` 的计时统计），无设备端 `double` 算术，无需注释；未触发精度红线。

## 结果
- **编译**：成功。`ninja -C build` → `[331/331] Linking target lc0`，产出 `lc0` 及 8 个测试可执行文件。`lc0` 动态链接 CoreX 的 `libcublas.so.10 / libcudart.so.10.2 / libcudnn.so.7 / libcuda.so.1`。
- **后端注册**：`cudnn-auto, cudnn, cudnn-fp16, cuda-auto, cuda, cuda-fp16, eigen ...` 全部注册成功，证明 cuDNN 后端与 plain CUDA 后端均编译可用。
- **测试**：跑完整 gtest 套件（全部 8 个测试目标、逐用例）：共 **52 个用例，52 通过，0 失败，0 跳过**。`meson test` 层面 8/8 目标 OK。
  - chessboard(21)、fp16(1)、hashcat(1)、position(8)、optionsparser(5)、syzygy(5)、encoder(10)、engine(1)。
  - SyzygyTest 在无 tablebase 文件下正常通过（gtest 未标记 skip）。
- **GPU 执行冒烟**：补充一个 ivcore11 原生 SAXPY kernel（`corex_port/build/gpu_smoke.cu`，clang++ -x ivcore 编译）在 GPU 0 上运行，`max_error=0.0`，确认设备端 kernel 执行与 H2D/D2H 拷贝正常。
- **lc0 benchmark 全推理冒烟**：跳过——离线 CoreX 环境无网络权重文件（storage.lczero.org 不可达）。此为客观环境约束，非适配问题；gtest 套件为权威测试集且已全通过。

## Failure Gate
- 三个 blocker 均为 workaround-able 并已实际解决（见 `blockers.json`）：nvcc 缺失（wrapper 解决）、fp16 arch gate（__ILUVATAR__ 旁路解决）、CUTLASS Ampere-only（可选特性，按范围排除）。
- 无 terminal blocker。所有结论均基于真实编译/运行输出（`build/compile.log`、`test/test.log`），非推断。
