# vla_jeston：基于 llama.cpp 的 π0.5 端侧 VLA 部署与优化

> 当前状态：项目规划与基线搭建阶段。
>
> 开发分支：`feature/pi05-deployment`。
>
> 本仓库以[官方 llama.cpp](https://github.com/ggml-org/llama.cpp)为运行时基础。EagleVLA-Edge 仅作为设计和实现参考，不作为项目依赖或最终交付物；OpenPI 作为数值正确性参考。

本项目的目标是在官方 llama.cpp、GGML 和 CUDA Backend 上实现一条可验证的 π0.5 端侧 VLA 推理链，并在 Jetson 上完成正确性闭环、性能分析和一项由 Profiling 驱动的 Runtime 优化。

项目不是简单复现或运行 EagleVLA-Edge，而是独立完成模型转换、计算图适配、持久化运行时、设备部署和实验验证。

## 一、端到端执行链

```text
双路相机图像 + 文本指令 + 机器人状态
                    |
                    v
          图像预处理与 Prompt 构造
                    |
                    v
      Vision Encoder + Multimodal Projector
                    |
                    v
           π0.5 Backbone Prefix
                    |
                    v
           Cross-Attention KV Cache
                    |
                    v
      Action Noise + State + Flow Time
                    |
                    v
       Action Expert + Flow Matching
                    |
                    v
            Continuous Action Chunk
                    |
                    v
          离线验证或机器人控制接口
```

## 二、七层开发结构

### 第一层：模型转换与 GGUF

将 π0.5 checkpoint 转换为 llama.cpp 可以加载的 GGUF 格式。

主要任务：

- 定义 π0.5 GGUF architecture 和模型元数据。
- 将原始 checkpoint 张量名映射为稳定的 GGUF 张量名。
- 写入 Action 维度、Action Chunk 长度等必要参数。
- 转换策略模型、Vision Encoder 和 Multimodal Projector。
- 校验张量名称、形状、数据类型、模型版本和文件 hash。

预计涉及：

```text
conversion/models/pi05.py
gguf-py/gguf/constants.py
gguf-py/gguf/tensor_mapping.py
```

第一道 Gate 是模型加载正确性：转换后的模型必须能够成功加载，并且元数据和张量清单与源 checkpoint 一致，然后才能进入计算图开发。

### 第二层：π0.5 模型计算图

在 llama.cpp 中注册 π0.5 模型架构，并实现策略模型的 GGML Graph。

主要任务：

- 构建多模态 Backbone 和 Prefix Graph。
- 构建 Action Expert Graph。
- 实现 Action Noise Projection 和 Flow Time Conditioning。
- 尽量复用现有 Attention、RoPE、RMSNorm、MLP 和 Backend 算子。
- 除非确实需要修改核心能力，否则将 π0.5 专用编排逻辑留在模型层和工具层。

预计涉及：

```text
src/llama-arch.h
src/llama-arch.cpp
src/llama-model-loader.cpp
src/models/pi05.cpp
```

### 第三层：多模态输入

扩展 llama.cpp 已有的 `mtmd` 基础设施，使其支持 π0.5 的输入协议。

主要任务：

- 按照 OpenPI 规则处理两路相机图像的 resize、crop 和 normalize。
- 保持正确的图像 Token 顺序和位置编码。
- 使用与参考实现一致的文本指令模板。
- 将 Vision 特征投影到策略 Backbone。
- 优先扩展已有 Vision Encoder 和 Preprocessor，不重复实现相同能力。

预计涉及：

```text
tools/mtmd/
```

### 第四层：持久化 VLA Runtime

在 llama.cpp 上方增加一个边界清晰的策略运行时，而不是把连续动作当作文本 Token 生成。

主要任务：

- 模型权重只加载一次。
- 将 Observation Encoding 与 Action Generation 分开。
- 在输入有效时复用 Prefix 和 Cross-Attention 状态。
- 复用 GGML Graph、Backend Buffer 和 Flow Matching Workspace。
- 通过小型接口暴露 Action Shape、推理结果和错误信息。
- 防止图像、Prompt 或机器人状态变化时错误复用旧缓存。

计划目录：

```text
tools/pi05/
├── CMakeLists.txt
├── pi05-run.cpp
├── pi05-session.cpp
├── pi05-session.h
└── pi05-bench.cpp
```

### 第五层：Jetson 部署

将设备配置、构建和测量逻辑与模型实现分离。

主要任务：

- 根据实际 JetPack 环境配置 ARM64 和 CUDA 构建。
- 检查模型文件、磁盘空间、内存容量和运行时依赖。
- 提供可复现的构建、部署和运行脚本。
- 采集 `tegrastats`、延迟、温度、功耗和内存数据。
- 固定 Jetson 型号、JetPack、CUDA、编译器、源码 commit 和模型 hash。

计划目录：

```text
deploy/jetson/
├── build.sh
├── configure.sh
├── run_pi05.sh
├── collect_env.sh
└── tegrastats.sh
```

### 第六层：正确性与性能验证

使用固定输入和 OpenPI 输出，在优化之前验证每一个重要边界。

数值检查顺序：

1. 图像预处理输出。
2. Vision Encoder 输出。
3. Prefix Hidden State。
4. Cross-Attention KV Tensor。
5. 单次 Action Expert 输出。
6. 每一轮 Flow Matching 更新结果。
7. 最终连续 Action Chunk。

持久化 Runtime 必须执行 `A -> B -> A` 等跨请求回归，检测非法缓存复用。

性能证据包括：

- 冷启动和稳态延迟。
- Vision、Prefix、Action Expert 分阶段耗时。
- 单轮 Flow Step 耗时。
- 端到端 P50/P95 延迟。
- 峰值内存、功耗、温度和持续运行频率。

计划目录：

```text
tests/pi05/
benchmarks/pi05/
```

### 第七层：Runtime 优化与 Speculative Flow 实验

只有在数值和端到端 Gate 通过后，才能根据 Profiling 结果选择优化点。

候选方向：

- 跨 Flow Step 复用 Prefix 和 Cross-Attention KV。
- 优化 GGML Graph 和 Backend Buffer 生命周期。
- 避免不必要的 Graph 重建、数据拷贝和同步。
- 重叠 CPU 图像预处理与 GPU 推理。
- 评估 Flow Step Unroll 和 CUDA Graph Capture。
- 只有 Profiling 证明必要时，才增加最小的 GGML 或 CUDA 改动。

π0.5 通过 Flow Matching 生成连续动作，不能直接套用 llama.cpp 的 Token Speculative Decoding。

在主线闭环后，可以研究 Speculative Flow：使用低成本 Draft Integrator 预测后续 Flow 状态，再由主 Action Expert 验证、校正或拒绝。该实验必须报告 Action 误差，不能把简单减少 Flow Step 描述为无损投机解码。

计划目录：

```text
experiments/speculative-flow/
```

## 三、开发阶段与验收 Gate

| 阶段 | 目标 | 完成条件 |
| --- | --- | --- |
| P0 | 固定 checkpoint 与输入输出协议 | 记录模型版本、张量形状、预处理、状态约定、初始噪声和 Flow 步数 |
| P1 | 构建官方 llama.cpp 基线 | 不含 π0.5 改动的 CPU 构建和可用 CUDA 主机构建通过 |
| P2 | 转换并加载 π0.5 GGUF | GGUF 元数据和张量清单与源 checkpoint 一致 |
| P3 | 实现 Vision 和 Prefix Graph | 中间输出在预先确定的容差内对齐 OpenPI |
| P4 | 实现 Action Expert 和 Flow Matching | 每轮 Flow 输出和最终 Action 通过数值检查 |
| P5 | 实现端到端持久化推理 | 重复请求、动态输入和 `A -> B -> A` 回归通过，内存保持稳定 |
| P6 | 建立 Jetson 基线 | 保存真实设备 Action、P50/P95、内存、功耗和原始日志 |
| P7 | 优化一个实际瓶颈 | 可回退源码改动通过同设备正确性和 A/B 性能回归 |
| P8 | 评估 Speculative Flow | 单独报告准确性与延迟权衡，不替代部署主线结果 |

## 四、计划中的项目目录

```text
vla_jeston/
├── conversion/                  # π0.5 -> GGUF 转换
├── gguf-py/                     # Architecture 元数据和 Tensor Mapping
├── src/models/pi05.cpp          # π0.5 GGML Graph
├── tools/mtmd/                  # 多模态输入适配
├── tools/pi05/                  # 持久化 VLA Runtime
├── tests/pi05/                  # 数值和 Session 回归
├── deploy/jetson/               # Jetson 构建和运行脚本
├── benchmarks/pi05/             # 性能测试与结果格式
├── experiments/speculative-flow/
└── docs/pi05/                   # 设计决策和复现文档
```

以上目录表示目标架构，仅在对应阶段开始时创建。只有完成数值 Gate 和真实 Jetson 设备 Gate 后，项目才会声明已经支持 π0.5 部署。

## 五、项目边界

本项目包含：

- π0.5 GGUF 转换和加载。
- Vision、Prefix、Action Expert 与 Flow Matching 计算图。
- 持久化 VLA Runtime。
- Jetson 部署、Profiling 和性能优化。
- OpenPI 数值对齐与跨请求回归。
- 主线完成后的 Speculative Flow 实验。

本项目不包含：

- π0.5 重新训练或强化学习训练。
- 机器人策略调参和真实任务成功率优化。
- 从零重写 llama.cpp、GGML 或 CUDA Backend。
- 将 EagleVLA-Edge 整体复制进本仓库。
- 在没有正确性证据的情况下用减少 Flow Step 制造加速结果。

## 六、参考项目

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：本项目的 Runtime 与 GGML/CUDA 基础。
- [EagleVLA-Edge](https://github.com/PKU-SEC-Lab/EagleVLA-Edge)：π0/π0.5 llama.cpp 部署思路参考。
- [OpenPI](https://github.com/Physical-Intelligence/openpi)：π0.5 模型语义与数值参考。

所有个人开发成果必须通过独立代码 diff、固定输入数值对齐、同设备性能对照和可复现日志证明；上游已有能力不得记录为个人实现。
