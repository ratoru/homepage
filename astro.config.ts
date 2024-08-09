import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import react from "@astrojs/react";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import sitemap from "@astrojs/sitemap";
import { transformerNotationDiff, transformerNotationFocus, transformerNotationHighlight } from "@shikijs/transformers";
import { SITE } from "./src/config";

import markdoc from "@astrojs/markdoc";

// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  integrations: [tailwind({
    applyBaseStyles: false
  }), react(), sitemap(), markdoc()],
  markdown: {
    remarkPlugins: [remarkToc, [remarkCollapse, {
      test: "Table of contents"
    }]],
    shikiConfig: {
      // For more themes, visit https://shiki.style/themes
      themes: {
        light: "rose-pine-dawn",
        dark: "one-dark-pro"
      },
      wrap: true,
      transformers: [transformerNotationDiff(), transformerNotationFocus(), transformerNotationHighlight()]
    }
  },
  vite: {
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"]
    }
  },
  scopedStyleStrategy: "where"
});