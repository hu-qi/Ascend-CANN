// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightDocSearch from "@astrojs/starlight-docsearch";
import sitemap from "@astrojs/sitemap";
import { loadEnv } from "vite";
import mermaid from 'astro-mermaid';

const { DOCSEARCH_APP_ID, DOCSEARCH_API_KEY, DOCSEARCH_INDEX_NAME } = loadEnv(
  process.env.NODE_ENV || "development",
  process.cwd(),
  ""
);

const enableDocSearch =
  DOCSEARCH_APP_ID && DOCSEARCH_API_KEY && DOCSEARCH_INDEX_NAME;
const getDocsearchPlugin = () =>
  starlightDocSearch({
    appId: DOCSEARCH_APP_ID,
    apiKey: DOCSEARCH_API_KEY,
    indexName: DOCSEARCH_INDEX_NAME,
  });

// https://astro.build/config
export default defineConfig({
  site: "https://ascend-cann.vercel.app",
  integrations: [
    mermaid({
      theme: 'forest',
      autoTheme: true
    }),
    sitemap(),
    starlight({
      title: "Ascend CANN",
      logo: {
        src: "./src/assets/my-logo.svg",
        replacesTitle: true,
      },
      social: [
        {
          icon: "github",
          label: "GitCode",
          href: "https://gitcode.com/nutpi/ascend-cann",
        },
      ],
      sidebar: [
        {
          label: "卷一 · 架构与开发",
          items: [
            { label: "第1篇：昇腾CANN算子库全景概览", slug: "ch01/第01篇-昇腾cann算子库全景概览" },
            { label: "第2篇：CANN算子开发通用架构与工程实践", slug: "ch01/第02篇-cann算子开发通用架构与工程实践" },
          ],
        },
        {
          label: "卷二 · 算子库专题",
          items: [
            { label: "第3篇（上）：ops-transformer", slug: "ch02/第03篇-ops-transformer上-transformer模型的高性能算子实现" },
            { label: "第3篇（下）：ops-transformer", slug: "ch02/第03篇-ops-transformer下-transformer模型的高性能算子实现" },
            { label: "第4篇（上）：ops-nn", slug: "ch02/第04篇-ops-nn上-神经网络基础算子的优化艺术" },
            { label: "第4篇（下）：ops-nn", slug: "ch02/第04篇-ops-nn下-神经网络基础算子的优化艺术" },
            { label: "第5篇：ops-math", slug: "ch02/第05篇-ops-math-数学运算的极致优化" },
            { label: "第6篇（上）：ops-cv", slug: "ch02/第06篇-ops-cv上-计算机视觉算子的硬件加速" },
            { label: "第6篇（下）：ops-cv", slug: "ch02/第06篇-ops-cv下-计算机视觉算子的硬件加速" },
            { label: "第7篇（上）：atvoss", slug: "ch02/第07篇-atvoss上-ai-vision优化的模板化编程模型" },
            { label: "第7篇（下）：atvoss", slug: "ch02/第07篇-atvoss下-ai-vision优化的模板化编程模型" },
          ],
        },
        {
          label: "卷三 · 性能与工具",
          items: [
            { label: "第8篇：Tiling机制与内存管理深度解析", slug: "ch03/第08篇-tiling机制与内存管理深度解析" },
            { label: "第9篇：异步编程与并行计算在CANN中的应用", slug: "ch03/第09篇-异步编程与并行计算在cann中的应用" },
            { label: "第10篇：量化技术与混合精度计算实践", slug: "ch03/第10篇-量化技术与混合精度计算实践" },
            { label: "第11篇：算子性能调优与实战指南", slug: "ch03/第11篇-算子性能调优与实战指南" },
          ],
        },
      ],
      plugins: enableDocSearch ? [getDocsearchPlugin()] : [],
      
    })
  ]
});
