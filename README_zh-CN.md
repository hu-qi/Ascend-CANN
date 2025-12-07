# Starlight 入门套件：基础

[![Built with Starlight](https://astro.badg.es/v2/built-with-starlight/tiny.svg)](https://starlight.astro.build)

```
pnpm create astro@latest -- --template starlight
```

> 🧑‍🚀 **经验丰富的宇航员？** 删除此文件。玩得开心！

## 🚀 项目结构

在您的 Astro + Starlight 项目中，您将看到以下文件夹和文件：

```
.
├── public/
├── src/
│   ├── assets/
│   ├── content/
│   │   └── docs/
│   └── content.config.ts
├── astro.config.mjs
├── package.json
└── tsconfig.json
```

Starlight 在 `src/content/docs/` 目录下查找 `.md` 或 `.mdx` 文件。每个文件根据其文件名作为一个路由公开。

图片可以添加到 `src/assets/` 并通过相对链接嵌入到 Markdown 中。

静态资源（如 favicon）可以放置在 `public/` 目录中。

## 🧞 命令

所有命令都在项目根目录下，通过终端运行：

| 命令                   | 作用                                         |
| :--------------------- | :------------------------------------------- |
| `pnpm install`         | 安装依赖                                     |
| `pnpm dev`             | 在 `localhost:4321` 启动本地开发服务器       |
| `pnpm build`           | 构建生产站点到 `./dist/`                     |
| `pnpm preview`         | 在部署前本地预览构建                         |
| `pnpm astro ...`       | 运行 CLI 命令，如 `astro add`、`astro check` |
| `pnpm astro -- --help` | 获取 Astro CLI 的帮助                        |

## 👀 想了解更多？

查看 [Starlight 文档](https://starlight.astro.build/)，阅读 [Astro 文档](https://docs.astro.build)，或加入 [Astro Discord 服务器](https://astro.build/chat)。
