import { defineMarkdocConfig, component } from "@astrojs/markdoc/config";
import shiki from "@astrojs/markdoc/shiki";

export default defineMarkdocConfig({
  extends: [
    shiki({
      // Choose from Shiki's built-in themes (or add your own)
      // Default: 'github-dark'
      // https://github.com/shikijs/shiki/blob/main/docs/themes.md
      theme: "one-dark-pro",
      // Enable word wrap to prevent horizontal scrolling
      // Default: false
      wrap: false,
      // Pass custom languages
      // Note: Shiki has countless langs built-in, including `.astro`!
      // https://github.com/shikijs/shiki/blob/main/docs/languages.md
      langs: [],
    }),
  ],
  tags: {
    aside: {
      render: component("./src/components/posts/Note.astro"),
    },
    footnote: {
      render: component("./src/components/posts/Footnote.astro"),
      attributes: {
        idNumber: { type: Number },
        label: { type: String },
      },
    },
    math: {
      render: component("./src/components/posts/Math.astro"),
      attributes: {
        formula: { type: String },
      },
    },
  },
});
