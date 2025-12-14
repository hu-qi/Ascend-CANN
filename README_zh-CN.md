# Ascend CANN 技术专题

Ascend CANN 算子库与性能优化系列的非官方整理，使用 Astro Starlight 构建。在线阅读：<https://ascend-cann.vercel.app>

## 文档目录

### 卷一 · 架构与开发
- [第1篇：昇腾CANN算子库全景概览](src/content/docs/ch01/第01篇-昇腾CANN算子库全景概览.md)：分层架构、算子库组成与生态定位。
- [第2篇：CANN算子开发通用架构与工程实践](src/content/docs/ch01/第02篇-CANN算子开发通用架构与工程实践.md)：Ascend C 编程模型、流水线并行与工程化范式。

### 卷二 · 算子库专题
- [第3篇（上）：ops-transformer](src/content/docs/ch02/第03篇-ops-transformer(上)-transformer模型的高性能算子实现.md)：GMM、FIA、Dispatch/Combine 等 Transformer 关键算子。
- [第3篇（下）：ops-transformer](src/content/docs/ch02/第03篇-ops-transformer(下)-transformer模型的高性能算子实现.md)：MoE、MLA、LLaMA/DeepSeek 优化与路由实现。
- [第4篇（上）：ops-nn](src/content/docs/ch02/第04篇-ops-nn(上)-神经网络基础算子的优化艺术.md)：基础神经网络算子分类、激活/卷积/归一化优化。
- [第4篇（下）：ops-nn](src/content/docs/ch02/第04篇-ops-nn(下)-神经网络基础算子的优化艺术.md)：池化、选择索引等算子实现与调优实践。
- [第5篇：ops-math](src/content/docs/ch02/第05篇-ops-math-数学运算的极致优化.md)：144 个数学算子的分类、精度与性能优化策略。
- [第6篇（上）：ops-cv](src/content/docs/ch02/第06篇-ops-cv(上)-计算机视觉算子的硬件加速.md)：图像几何变换、插值与目标检测前处理算子。
- [第6篇（下）：ops-cv](src/content/docs/ch02/第06篇-ops-cv(下)-计算机视觉算子的硬件加速.md)：NMS/IoU、实例分割、图像增强与 3D 视觉算子。
- [第7篇（上）：atvoss](src/content/docs/ch02/第07篇-atvoss(上)-ai-vision优化的模板化编程模型.md)：模板化编程框架架构、五层设计与编程范式。
- [第7篇（下）：atvoss](src/content/docs/ch02/第07篇-atvoss(下)-ai-vision优化的模板化编程模型.md)：高级特性、表达式模板、实践案例与调优技巧。

### 卷三 · 性能与工具
- [第8篇：Tiling机制与内存管理深度解析](src/content/docs/ch03/第08篇-Tiling机制与内存管理深度解析.md)：Tiling 框架、动态策略与内存层次优化。
- [第9篇：异步编程与并行计算在CANN中的应用](src/content/docs/ch03/第09篇-异步编程与并行计算在cann中的应用.md)：受限异步模型、Stream 调度与多级并行。
- [第10篇：量化技术与混合精度计算实践](src/content/docs/ch03/第10篇-量化技术与混合精度计算实践.md)：FP16/BF16/INT8 能力、低比特思路与混合精度最佳实践。
- [第11篇：算子性能调优与实战指南](src/content/docs/ch03/第11篇-算子性能调优与实战指南.md)：性能指标体系、瓶颈分析器与实战调优方法论。

## 本地开发
1. 安装 Node.js 18+ 与 pnpm（建议 `corepack enable pnpm`）。
2. 安装依赖：`pnpm install`
3. 本地预览：`pnpm dev`，默认端口 `4321`
4. 生产构建：`pnpm build`，产物位于 `dist/`
5. 本地预览产物：`pnpm preview`
6. 代码检查：`pnpm astro check`

## 项目结构
- `src/content/docs/ch01`：CANN 总览与算子开发架构
- `src/content/docs/ch02`：各算子库与模板化框架专题
- `src/content/docs/ch03`：性能优化、量化与并行编程
- `astro.config.mjs`：Starlight 站点配置与导航
- `src/assets`：站点 Logo 与插图

## 参与贡献
- 欢迎通过 Issue/PR 补充案例、修正文档或改进导航。
- 保持 Markdown 标题层级清晰，必要时同步更新 `astro.config.mjs` 的侧边栏配置。
