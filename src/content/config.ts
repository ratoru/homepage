import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  schema: z.object({
    title: z.string({
      required_error: "Required frontmatter missing: title",
      invalid_type_error: "title must be a string",
    }),
    description: z.optional(z.string()),
    tags: z.optional(z.array(z.string())),
    date: z.date({
      required_error: "Required frontmatter missing: date",
      invalid_type_error:
        "date must be written in yyyy-mm-dd format without quotes: For example, Jan 22, 2000 should be written as 2000-01-22.",
    }),
  }),
});

export const collections = { blog };
