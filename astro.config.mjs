// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightDocSearch from "@astrojs/starlight-docsearch";
import sitemap from "@astrojs/sitemap";
import { loadEnv } from "vite";

const { DOCSEARCH_APP_ID, DOCSEARCH_API_KEY, DOCSEARCH_INDEX_NAME } = loadEnv(
  process.env.NODE_ENV || "development",
  process.cwd(),
  ""
);

// https://astro.build/config
export default defineConfig({
  site: "https://ascend.nutpi.net",
  integrations: [
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
          href: "https://gitcode.com/nutpi/starlight",
        },
      ],
      sidebar: [
        {
          label: "指南",
          items: [
            // Each item here is one entry in the navigation menu.
            { label: "示例指南", slug: "guides/example" },
          ],
        },
        {
          label: "参考",
          autogenerate: { directory: "reference" },
        },
      ],
      plugins: [
        starlightDocSearch({
          appId: DOCSEARCH_APP_ID,
          apiKey: DOCSEARCH_API_KEY,
          indexName: DOCSEARCH_INDEX_NAME,
        }),
      ],
    }),
  ],
});
