#include "acl/acl.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>

// 引入自动生成的头文件
#include "aclrtlaunch_vector_add.h"

// 【关键点1】TilingData 结构体升级
// 必须与优化后的 Kernel 端定义严格一致 (顺序、类型都不能错)
struct TilingData {
    uint32_t totalLength;      // 总数据量
    uint32_t tileNum;          // 切分总次数 (循环次数)
    uint32_t tileLength;       // 单次搬运的标准长度
    uint32_t lastTileLength;   // 最后一次搬运的长度 (尾块)
    uint32_t startOffset;      // 起始偏移量
};

// 结果验证辅助函数
float CheckResult(uint16_t hexVal) {
    uint16_t sign = (hexVal >> 15) & 0x1;
    uint16_t exp = (hexVal >> 10) & 0x1F;
    uint16_t mantissa = hexVal & 0x3FF;
    if (exp == 0) return 0.0f;
    int exponent = (int)exp - 15;
    float m = 1.0f + (float)mantissa / 1024.0f;
    float val = m * std::pow(2.0f, exponent);
    return (sign == 0) ? val : -val;
}

int main() {
    // 1. 定义基础参数
    int32_t deviceId = 0;
    
    // 【场景设定】
    // 总长度 8192。为了验证尾块逻辑，我们可以故意设置一个
    // 不能被 tileLength 整除的数，或者直接用整除的数验证基础功能。
    // 这里使用 8192 (8KB FP16)
    uint32_t totalLength = 8192; 
    
    // 【关键点2】Host 端计算 Tiling 策略
    // 假设我们使用 1 个核来处理这 8192 个数据
    // 注意：优化后的 Kernel 读取 startOffset。如果我们启动多个核，但只传一个 TilingData，
    // 所有核都会处理同一段内存导致冲突。简单起见，这里演示单核逻辑。
    uint32_t blockDim = 1; 

    // 设定 UB 上一次处理的数据量 (Tile Length)
    // 假设 UB 每次处理 128 个 half (FP16) 元素 = 256 Bytes
    // 这个值通常根据 UB 的物理大小 (如 192KB) 来最大化利用
    uint32_t tileLength = 128; 

    // 计算循环次数 (tileNum) = ceil(totalLength / tileLength)
    // 算法： (A + B - 1) / B 实现向上取整
    uint32_t tileNum = (totalLength + tileLength - 1) / tileLength;

    // 计算尾块长度 (lastTileLength)
    // 如果恰好整除，尾块长度等于标准长度；否则等于余数
    uint32_t lastTileLength = totalLength % tileLength;
    if (lastTileLength == 0) lastTileLength = tileLength;
    
    // 计算起始偏移 (单核模式下为0)
    uint32_t startOffset = 0;

    // 打印计算好的 Tiling 策略，方便调试
    std::cout << "--- Tiling Strategy Calculated on Host ---" << std::endl;
    std::cout << "Total Length:   " << totalLength << std::endl;
    std::cout << "Tile Length:    " << tileLength << std::endl;
    std::cout << "Tile Num(Loop): " << tileNum << std::endl;
    std::cout << "Last Tile Len:  " << lastTileLength << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // 2. 准备数据结构
    TilingData hostTiling = {totalLength, tileNum, tileLength, lastTileLength, startOffset};
    uint32_t dataSize = totalLength * sizeof(uint16_t);
    uint32_t tilingSize = sizeof(TilingData);

    // 3. ACL 初始化
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 4. Host 数据准备 (1.0 + 2.0)
    std::vector<uint16_t> hostX(totalLength, 0x3C00); // 1.0
    std::vector<uint16_t> hostY(totalLength, 0x4000); // 2.0
    std::vector<uint16_t> hostZ(totalLength, 0);

    // 5. Device 内存分配
    void *devX, *devY, *devZ, *devTiling;
    aclrtMalloc(&devX, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devZ, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devTiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 数据拷贝 (Input & Tiling)
    aclrtMemcpy(devX, dataSize, hostX.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devY, dataSize, hostY.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devTiling, tilingSize, &hostTiling, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 7. 启动核函数
    // 使用算好的 blockDim (1)
    ACLRT_LAUNCH_KERNEL(vector_add)(blockDim, stream, devX, devY, devZ, nullptr, devTiling);

    // 8. 同步与回传
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostZ.data(), dataSize, devZ, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 9. 验证结果
    std::cout << "Run Finish!" << std::endl;
    
    // 验证第 0 个数据 (标准块内)
    float val0 = CheckResult(hostZ[0]);
    std::cout << "Index[0] Value: " << val0 << " (Expected: 3.0)" << std::endl;

    // 验证最后一个数据 (验证尾块处理是否正确)
    float valLast = CheckResult(hostZ[totalLength - 1]);
    std::cout << "Index[" << totalLength-1 << "] Value: " << valLast << " (Expected: 3.0)" << std::endl;

    if (val0 == 3.0f && valLast == 3.0f) {
        std::cout << "TEST PASSED!" << std::endl;
    } else {
        std::cout << "TEST FAILED." << std::endl;
    }

    // 10. 资源释放
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devZ); aclrtFree(devTiling);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}