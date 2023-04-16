import { defineMarkdocConfig } from "@astrojs/markdoc/config";
import Aside from "./src/components/posts/Aside.astro";
import CodeBlock from "./src/components/posts/CodeBlock.astro";

export default defineMarkdocConfig({
  tags: {
    aside: {
      render: Aside,
      attributes: {
        type: { type: String },
        title: { type: String },
      },
    },
  },
  nodes: {
    fence: {
      render: CodeBlock,
      attributes: {
        content: { type: String },
        language: { type: String },
        process: { type: Boolean },
      },
    },
  },
});
