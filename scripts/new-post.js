#!/usr/bin/env node

// Usage: node scripts/new-post.js
// Or via npm: pnpm new-post

import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";

const rl = createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise((res) => rl.question(q, res));

function slugify(title) {
	return title
		.toLowerCase()
		.replace(/[^\w\s-]/g, "")
		.trim()
		.replace(/[\s_]+/g, "-")
		.replace(/-+/g, "-");
}

const title = (await ask("Title: ")).trim();
if (!title) {
	console.error("Title is required.");
	process.exit(1);
}

const description = (await ask("Description: ")).trim();
const tagsInput = (await ask("Tags (comma-separated, optional): ")).trim();
const draft = (await ask("Draft? (y/N): ")).trim().toLowerCase() === "y";

rl.close();

const today = new Date().toISOString().split("T")[0];
const slug = slugify(title);
const tags = tagsInput
	? tagsInput
			.split(",")
			.map((t) => t.trim().toLowerCase())
			.filter(Boolean)
	: [];

const tagsYaml = tags.length > 0 ? `\ntags: [${tags.map((t) => `"${t}"`).join(", ")}]` : "";
const draftYaml = draft ? "\ndraft: true" : "";

const frontmatter = `---
title: "${title}"
description: "${description}"
publishDate: ${today}${tagsYaml}${draftYaml}
---

`;

const filename = `${slug}.md`;
const filepath = join(import.meta.dirname, "../src/content/post", filename);

if (existsSync(filepath)) {
	console.error(`File already exists: ${filepath}`);
	process.exit(1);
}

writeFileSync(filepath, frontmatter);
console.log(`Created: src/content/post/${filename}`);
