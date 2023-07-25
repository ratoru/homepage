import { defineMarkdocConfig, component } from "@astrojs/markdoc/config";

export default defineMarkdocConfig({
  tags: {
    aside: {
      // render: Aside,
      render: component('./src/components/posts/Aside.astro'),
      attributes: {
        type: { type: String },
        title: { type: String },
      },
    },
  },
  nodes: {
    fence: {
      // render: CodeBlock,
      render: component('./src/components/posts/CodeBlock.astro'),
      attributes: {
        content: { type: String },
        language: { type: String },
        process: { type: Boolean },
      },
    },
  },
});
