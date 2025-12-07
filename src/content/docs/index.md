---
title: Ascend CANN 知识库
description: 昇腾 CANN 算子库架构、算子实现与性能优化的专题整理。
template: splash
hero:
  tagline: 11 篇文章拆解 Ascend CANN 算子栈
  image:
    file: ../../assets/my-logo.png
  actions:
    - text: 从全景概览开始
      link: /ch01/第01篇-昇腾CANN算子库全景概览/
      icon: right-arrow
    - text: 直接看性能调优
      link: /ch03/第11篇-算子性能调优与实战指南/
      icon: book
    - text: 在线站点
      link: https://ascend.nutpi.net
      icon: external
      variant: minimal
---

## 关于项目
- 非官方的 Ascend CANN 算子库技术专题，涵盖架构、算子开发、性能优化等核心内容。
- 基于 Astro Starlight 构建，支持 Algolia DocSearch 与 sitemap，便于索引与检索。
- 文章同步托管于仓库，可本地启动后离线阅读。

## 目录
**卷一 · 架构与开发**
- [第1篇：昇腾CANN算子库全景概览](/ch01/第01篇-昇腾CANN算子库全景概览/)：分层架构、算子库组成与生态定位。
- [第2篇：CANN算子开发通用架构与工程实践](/ch01/第02篇-CANN算子开发通用架构与工程实践/)：Ascend C 编程模型、流水线并行与工程化范式。

**卷二 · 算子库专题**
- [第3篇（上）：ops-transformer](/ch02/第03篇-ops-transformer(上)-Transformer模型的高性能算子实现/)：GMM、FIA、Dispatch/Combine 等 Transformer 关键算子。
- [第3篇（下）：ops-transformer](/ch02/第03篇-ops-transformer(下)-Transformer模型的高性能算子实现/)：MoE、MLA、LLaMA/DeepSeek 优化与路由实现。
- [第4篇（上）：ops-nn](/ch02/第04篇-ops-nn(上)-神经网络基础算子的优化艺术/)：基础神经网络算子分类、激活/卷积/归一化优化。
- [第4篇（下）：ops-nn](/ch02/第04篇-ops-nn(下)-神经网络基础算子的优化艺术/)：池化、选择索引等算子实现与调优实践。
- [第5篇：ops-math](/ch02/第05篇-ops-math-数学运算的极致优化/)：144 个数学算子的分类、精度与性能优化策略。
- [第6篇（上）：ops-cv](/ch02/第06篇-ops-cv(上)-计算机视觉算子的硬件加速/)：图像几何变换、插值与目标检测前处理算子。
- [第6篇（下）：ops-cv](/ch02/第06篇-ops-cv(下)-计算机视觉算子的硬件加速/)：NMS/IoU、实例分割、图像增强与 3D 视觉算子。
- [第7篇（上）：atvoss](/ch02/第07篇-atvoss(上)-AI Vision优化的模板化编程模型/)：模板化编程框架架构、五层设计与编程范式。
- [第7篇（下）：atvoss](/ch02/第07篇-atvoss(下)-AI Vision优化的模板化编程模型/)：高级特性、表达式模板、实践案例与调优技巧。

**卷三 · 性能与工具**
- [第8篇：Tiling机制与内存管理深度解析](/ch03/第08篇-Tiling机制与内存管理深度解析/)：Tiling 框架、动态策略与内存层次优化。
- [第9篇：异步编程与并行计算在CANN中的应用](/ch03/第09篇-异步编程与并行计算在CANN中的应用/)：受限异步模型、Stream 调度与多级并行。
- [第10篇：量化技术与混合精度计算实践](/ch03/第10篇-量化技术与混合精度计算实践/)：FP16/BF16/INT8 能力、低比特思路与混合精度最佳实践。
- [第11篇：算子性能调优与实战指南](/ch03/第11篇-算子性能调优与实战指南/)：性能指标体系、瓶颈分析器与实战调优方法论。

## 本地预览
- 克隆代码后执行 `pnpm install && pnpm dev`，默认端口 `4321`。
- 调整侧边栏或导航请同步更新 `astro.config.mjs`，以保证目录一致性。
