import { defineMarkdocConfig, component } from "@astrojs/markdoc/config";
import shiki from "@astrojs/markdoc/shiki";
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationFocus,
} from "@shikijs/transformers";

export default defineMarkdocConfig({
  extends: [
    shiki({
      // For more themes, visit https://shiki.style/themes
      themes: {
        light: "rose-pine-dawn",
        dark: "one-dark-pro",
      },
      // Enable word wrap to prevent horizontal scrolling
      // Default: false
      wrap: true,
      // Pass custom languages
      // Note: Shiki has countless langs built-in, including `.astro`!
      // https://github.com/shikijs/shiki/blob/main/docs/languages.md
      langs: [],
      // https://shiki.style/packages/transformers
      transformers: [
        transformerNotationDiff(),
        transformerNotationHighlight(),
        transformerNotationFocus(),
      ],
    }),
  ],
  tags: {
    aside: {
      render: component("./src/components/markdoc/Note.astro"),
    },
    footnote: {
      render: component("./src/components/markdoc/Footnote.astro"),
      attributes: {
        idNumber: { type: Number },
        label: { type: String },
      },
    },
    math: {
      render: component("./src/components/markdoc/Math.astro"),
      attributes: {
        formula: { type: String },
      },
    },
  },
});
