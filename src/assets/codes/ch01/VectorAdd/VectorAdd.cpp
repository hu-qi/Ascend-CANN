/**
 * Ascend C 向量加法核函数实现
 * * 核心特性：
 * 1.流水线并行 (Pipeline Parallelism): CopyIn -> Compute -> CopyOut
 * 2.双缓冲 (Double Buffer): 使用两个缓冲区掩盖数据搬运延迟
 * 3.Host-Device Tiling: 切分策略在 Host 端计算，Kernel 端直接执行，减少标量计算开销
 * 4.动态尾块处理: 支持非 32Byte 对齐或非标准块大小的数据长度
 */

#include "kernel_operator.h"

using namespace AscendC;

// -------------------------------------------------------------------------
// 1. Tiling 数据结构定义
// -------------------------------------------------------------------------
// 定义 Tiling 结构体，用于从 Host 端接收计算好的切分参数
// 【注意】结构体的内存布局（成员顺序、类型）必须与 Host 端代码完全一致
struct TilingData {
    uint32_t totalLength;      // 总数据量 (仅用于参考或边界检查)
    uint32_t tileNum;          // 总循环次数 (Block 内需要切分多少次)
    uint32_t tileLength;       // 标准切分长度 (UB 缓冲区的最大分配大小)
    uint32_t lastTileLength;   // 最后一次搬运的长度 (用于处理除不尽的余数/尾块)
    uint32_t startOffset;      // 当前核处理数据的全局起始偏移 (支持多核非均匀切分)
};

// Tiling 解析辅助函数
// 将 Global Memory (显存) 上的 tiling 数据拷贝到 Local 变量中供 Kernel 使用
__aicore__ inline void InitTilingData(GM_ADDR tiling, TilingData* tilingData) {
    __gm__ uint32_t* tilingGM = (__gm__ uint32_t*)tiling;
    tilingData->totalLength    = tilingGM[0];
    tilingData->tileNum        = tilingGM[1];
    tilingData->tileLength     = tilingGM[2];
    tilingData->lastTileLength = tilingGM[3];
    tilingData->startOffset    = tilingGM[4];
}

// -------------------------------------------------------------------------
// 2. 算子类实现
// -------------------------------------------------------------------------
class VectorAdd {
public:
    __aicore__ inline VectorAdd() {}

    /**
     * 初始化函数 (Init)
     * 职责：解析参数，设置内存地址，分配片上内存 (Pipe/Queue)
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, TilingData* tData) {
        // 保存调度所需的关键参数
        this->tileNum = tData->tileNum;
        this->tileLength = tData->tileLength;
        this->lastTileLength = tData->lastTileLength;

        // [内存地址设置]
        // 使用 Host 端传来的 startOffset 设置 GlobalTensor 的起始地址
        // 相比于在 Kernel 内用 GetBlockIdx 计算，这种方式更灵活，支持不均匀切分
        xGm.SetGlobalBuffer((__gm__ half*)x + tData->startOffset, tData->totalLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + tData->startOffset, tData->totalLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + tData->startOffset, tData->totalLength);

        // [管道初始化]
        // BUFFER_NUM = 2 开启双缓冲
        // allocSize 必须按照最大可能的块大小 (tileLength) 进行分配
        // 即使处理尾块时用不完这么多，也要按最大值预留
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    /**
     * 处理函数 (Process)
     * 职责：控制流水线循环调度
     */
    __aicore__ inline void Process() {
        // 循环处理每一个数据块 (Tile)
        for (int32_t i = 0; i < tileNum; i++) {
            // [尾块处理核心逻辑]
            // 如果是最后一次循环 (i == tileNum - 1)，使用 lastTileLength
            // 否则使用标准的 tileLength
            uint32_t currentLen = (i == tileNum - 1) ? lastTileLength : tileLength;

            // 依次启动流水线的三个阶段
            CopyIn(i, currentLen);
            Compute(i, currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    /**
     * 阶段 1: CopyIn (搬入)
     * 职责：GM (Global Memory) -> UB (Unified Buffer)
     */
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        // 1. 从队列申请空闲 Tensor (如果队列满则阻塞)
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

        // 2. 执行数据搬运 (DMA)
        // [偏移计算]: progress * tileLength (始终按标准块长跳跃)
        // [搬运长度]: length (如果是尾块，只搬运剩余部分)
        DataCopy(xLocal, xGm[progress * tileLength], length);
        DataCopy(yLocal, yGm[progress * tileLength], length);

        // 3. 入队，通知 Compute 阶段数据已准备好
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    /**
     * 阶段 2: Compute (计算)
     * 职责：UB -> UB (利用 Vector Unit 进行计算)
     */
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        // 1. 出队，获取输入数据 (如果队列空则阻塞)
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        
        // 2. 申请输出 Tensor
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        // 3. 执行向量加法
        // 只计算 length 长度的数据，避免计算无效数据
        Add(zLocal, xLocal, yLocal, length);

        // 4. 结果入队，通知 CopyOut 阶段
        outQueueZ.EnQue(zLocal);
        
        // 5. 释放输入 Tensor (实现双缓冲的关键，释放后 CopyIn 可复用这块内存)
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    /**
     * 阶段 3: CopyOut (搬出)
     * 职责：UB -> GM
     */
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        // 1. 出队，获取计算结果
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();

        // 2. 将结果搬回 Global Memory
        // 同样注意偏移量是 progress * tileLength，长度是 length
        DataCopy(zGm[progress * tileLength], zLocal, length);

        // 3. 释放输出 Tensor
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // [双缓冲配置]
    // 必须在 TQue 定义之前声明。2 代表队列深度为 2，允许 ping-pong 操作
    static constexpr int32_t BUFFER_NUM = 2;

    // 内存管理对象
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY; // 输入队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;         // 输出队列
    
    // 全局内存对象 (类似指针)
    GlobalTensor<half> xGm, yGm, zGm;

    // 成员变量
    uint32_t tileNum;        // 循环次数
    uint32_t tileLength;     // 标准块长
    uint32_t lastTileLength; // 尾块长
};

// -------------------------------------------------------------------------
// 3. 核函数入口
// -------------------------------------------------------------------------
extern "C" __global__ __aicore__ void vector_add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    // 1. 解析 Host 传入的 Tiling 参数
    // 数据存放在 Scalar Buffer (栈) 上
    TilingData tilingData;
    InitTilingData(tiling, &tilingData);

    // 2. 创建算子实例
    VectorAdd op;
    
    // 3. 初始化并传入完整的 Tiling 结构体
    op.Init(x, y, z, &tilingData);
    
    // 4. 执行算子逻辑
    op.Process();
}