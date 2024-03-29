import { defineConfig } from "astro/config";
import markdoc from "@astrojs/markdoc";
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: 'https://ratoru.com',
  integrations: [markdoc(), tailwind()],
});
