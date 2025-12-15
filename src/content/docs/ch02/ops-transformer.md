# 🚀 爆款技术博客：小白也能看懂！手把手带你跑通 Ascend CANN 自定义算子

> **摘要**：还在为 AI 算子开发头秃吗？听说国产算力昇腾（Ascend）越来越火，想入门却不知从何下手？本文以 `ops-transformer` 算子库为例，带你深入浅出理解 CANN 算子开发的核心概念，并手把手教你跑通第一个自定义算子！文末更有 FlashAttention 等进阶干货，建议收藏！🌟

---

## 👋 写在前面

大家好，我是你们的昇腾技术向导。

随着大模型（LLM）时代的到来，**Transformer** 架构成为了绝对的王者。而要让这些庞然大物在硬件上跑得飞快，离不开底层**算子**（Operator）的极致优化。

今天通过深扒 `ascend-cann/ops/ops-transformer` 这个开源项目，带大家揭开 CANN 算子开发的神秘面纱。不管你是刚接触 AI 编译器的萌新，还是想转行做底层优化的同学，这篇文章都能带你入门！

---

## 🧐 核心概念：Ascend 与 CANN

在开始撸代码之前，我们先统一一下“语言”：

1.  **Ascend（昇腾）**：华为推出的 AI 处理器（NPU），类似于大家熟悉的 GPU，但专门为 AI 计算设计。
2.  **CANN (Compute Architecture for Neural Networks)**：昇腾的软件栈，对应到 NVIDIA 那边就是 CUDA。它是连接上层框架（PyTorch/MindSpore）和底层硬件的桥梁。
3.  **Host 与 Device**：
    - **Host 侧**：通常指 CPU，负责逻辑控制、资源管理、数据预处理。
    - **Device 侧**：指 NPU（AI Core），负责“大力出奇迹”，处理密集的矩阵计算。

**算子开发的核心逻辑**：在 Host 侧准备好数据和任务，扔给 Device 侧去狂算，算完再把结果拿回来。

---

## 🛠️ 实战演练：从 "Hello World" (Add 算子) 开始

`ops-transformer` 库里虽然有 FlashAttention 这种“核弹级”算子，但为了不把大家劝退，我们先看一个最简单的例子：**Add 算子**（两个张量相加）。

位置指路：`examples/add_example`

### 1. 看看 Device 侧代码 (Kernel)

这是算子的核心算法，运行在 NPU 的 AI Core 上。

文件：`examples/add_example/op_kernel/add_example.cpp`

```cpp
// 引入必要的头文件
#include "add_example.h"

// 算子入口函数
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 1. 获取 Tiling 数据 (切分策略)
    // Tiling 是 CANN 开发的精髓，因为 NPU 内存有限，需要把大矩阵切成小块分批计算
    REGISTER_TILING_DEFAULT(AddExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);

    // 2. 根据不同的模式 (float 或 int32) 初始化并执行
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsAddExample::AddExample<float> op;
        op.Init(x, y, z, &tilingData);      // 初始化
        op.Process();                       // 执行计算
    }
    // ... int32 处理逻辑 ...
}
```

**划重点：**

- `__global__ __aicore__`：告诉编译器，这玩意儿是在 AI Core 上跑的。
- `Tiling`：因为 AI Core 的片上内存（L1/UB）很小，放不下整个大模型的数据，所以必须**分块（Tile）**。Host 侧算出怎么切，Device 侧拿着这个“切分说明书”去搬运和计算数据。

### 2. 看看 Host 侧代码 (调用)

Host 侧主要负责：**初始化设备 -> 搬运数据 -> 发射核函数 -> 同步结果**。

我们使用 **ACLNN (Ascend Computing Language for Neural Networks)** 接口来调用，这是目前推崇的高级接口。

文件：`examples/add_example/examples/test_aclnn_add_example.cpp` (精简版)

```cpp
int main() {
    // 1. 初始化设备 (Device 0)
    int32_t deviceId = 0;
    aclrtStream stream;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    // 2. 构造数据 (这里省略了造数据的过程，假设 selfX, selfY 都是 32x4x4x4 的 Tensor)
    // ... CreateAclTensor ...

    // 3. 计算 Workspace 大小 (显存除了存数据，还需要存中间变量)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);

    // 4. 申请 Workspace 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 5. 执行算子！(异步执行，放入 Stream 队列)
    aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);

    // 6. 同步等待 (Host 等 Device 算完)
    aclrtSynchronizeStream(stream);

    // 7. 打印结果 & 释放资源
    PrintOutResult(outShape, &outDeviceAddr);
    // ... aclDestroyTensor, aclrtFree ...

    return 0;
}
```

**流程总结：**

1.  `GetWorkspaceSize`：问一下底层，“大哥，这一票你要用多少缓冲区？”
2.  `aclrtMalloc`：给大哥分配缓冲区。
3.  `aclnnAddExample`：开搞！
4.  `Synchronize`：收工。

---

## 🏃‍♂️ 动起手来：怎么运行？

只要你有 CANN 环境（通常在服务器上的 `/usr/local/Ascend`），就可以按以下步骤起飞。

### 第一步：编译项目

在 `ops-transformer` 根目录下，通常有 `build.sh` 脚本。

```bash
# 进入项目目录
cd ops-transformer

# 运行编译脚本
bash build.sh
```

编译成功后，会在 `output` 或 `build` 目录下生成算子二进制包（Run package）。

### 第二步：运行示例

对于 `add_example`，通常需要独立编译它的测试用例。

```bash
cd examples/add_example
mkdir build && cd build
cmake ..
make

# 运行程序
./test_aclnn_add_example
```

如果你看到类似下面的输出，恭喜你！你已经成功调用了 NPU 算子！🎉

```text
mean result[0] is: 2.000000
mean result[1] is: 2.000000
...
```

_(因为 1+1=2，没毛病！)_

---

## 🔥 进阶：FlashAttention (核弹级优化)

学会了 `1+1`，我们看看真正的“工业界宠儿” —— **FlashAttention**。

在 `ops-transformer/attention/flash_attention_score` 中，你可以找到它的身影。

它的复杂程度呈指数级上升：

- **输入多**：Query, Key, Value, Mask, Dropout...
- **计算繁**：MatMul -> Softmax -> MatMul，中间还不能写回 HBM（显存），必须在极小的片上内存里“反复横跳”。
- **Tiling 难**：如何切分 Q/K/V 才能让流水线这辆“法拉利”不堵车？

但正因为难，它的性能是普通 Attention 的数倍！这也正是 CANN 算子工程师的高薪秘诀所在。😉

---

## 💀 硬核预警：手撕 FlashAttention 源码

既然大家都说 CANN 开发能挖掘硬件极致性能，那我们就拿 **FlashAttention** 开刀，看看在 `ops-transformer/attention/flash_attention_score` 底下，究竟藏着什么黑科技。

我们重点围观 `op_kernel/flash_attention_score_s1s2_bn2gs1.h` 这个文件，这是 FlashAttention 核心实现的一个模板类。

### 1. 极致的内存管理：Ping-Pong 双缓冲

在 NPU 开发中，最忌讳的就是 **“计算等数据”**。为了让 AI Core 一刻不停地连轴转，源码中大量使用了 **Double Buffering (双缓冲)** 机制，也就是常说的 Ping-Pong 流水线。

```cpp
// 源码片段：flash_attention_score_s1s2_bn2gs1.h
TBuf<> maskTBufPing;
TBuf<> maskTBufPong; // 掩码也搞双份，Ping负责算，Pong负责搬
TBuf<> stage1PingBuf;
TBuf<> stage1PongBuf;
// ...
```

**原理揭秘**：
当 `Ping` 缓冲区正在被 Vector 单元进行计算（比如 Softmax）时，MTE（Memory Transfer Engine）搬运单元同时在疯狂地把下一块数据搬到 `Pong` 缓冲区。这样计算和搬运在时间上完美重叠，流水线打满！

### 2. 这里的 C++ 会“变身”：模板元编程

你会发现代码里充斥着大量的 `template`：

```cpp
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, ...>
class FlashAttentionScoreS1s2Bn2gs1 { ... }
```

**为什么要写得这么“复杂”？**
这是为了**编译期优化**！
Host 侧在发射算子时，会根据数据类型（float16/bfloat16）、Layout（BNSD/BSH）等参数，实例化出最匹配的那个 Kernel。这样在 Device 侧运行时，就没有了 `if (isFloat) ... else ...` 的冗余判断，所有的路径在编译时就确定了，执行效率直接拉满。

### 3. Tiling 切分：化整为零

面对超长的 Sequence Length（比如 4096 甚至更长），显存根本放不下 $Q \times K^T$ 这么大的矩阵。FlashAttention 的精髓就是分块计算。

```cpp
// 源码片段：Process() 函数中
GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
// ...
for (int64_t bngIdx = bngStartIdx; bngIdx < bngEndIdx; ++bngIdx) {
    // 外层循环：遍历 Batch 和 Head
    // 内层循环：遍历 Q (Query) 的分块 (S1)
    // 最内层：遍历 K/V (Key/Value) 的分块 (S2)
}
```

代码通过 `GET_TILING_DATA` 拿到 Host 侧算好的切分策略，然后用多重循环，像切蛋糕一样，把巨大的 Attention 矩阵切成小块（Tile）。每一个小块计算完 Score 后，利用 Online Softmax 技巧实时更新结果，避免了存下整个 $N \times N$ 矩阵。

### 4. 算力核心：MatMul 抽象接口

CANN 提供了非常高层的 `matmul` 类，封装了底层的 Cube 单元指令。

```cpp
using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
// ...
matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, GetMmCfg(enableL1Reuse)> bmm1;
```

你不需要手写复杂的汇编，只需要定义好 `bmm1` (Batch MatMul 1) 的输入输出类型，然后调用 `bmm1.SetTensorA`, `bmm1.SetTensorB`, `bmm1.Iterate()`，底层的 Cube 单元就会轰隆隆地动起来，完成 $Q \times K^T$ 的矩阵乘法。

---

## 📝 总结

今天我们探索了 `ops-transformer` 库，从目录结构到代码实操，走通了 CANN 算子开发的最小闭环：

1.  **Ops-Transformer**：昇腾大模型算子宝藏库。
2.  **Add 算子**：麻雀虽小，五脏俱全，理解 Host/Device 交互的最佳案例。
3.  **开发流程**：`Tiling` (切分) -> `Kernel` (计算) -> `ACLNN` (调用)。

**下一步建议**：

- 下载代码，亲自跑通 `add_example`。
- 尝试修改 `add_example`，比如改成 `sub_example` (减法) 或 `mul_example` (乘法)，感受一下掌控硬件的快感。

关注我，下期带你手撕 **FlashAttention** 源码！🚀

---

_Github 链接：[ascend-cann/ops/ops-transformer](https://gitee.com/ascend/ops-transformer) (示例链接，实际请参考内部或官方仓库)_
