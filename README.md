# Ratoru Homepage

This is my personal website's 5th iteration. You can find my blog and my past work [there](https://ratoru.com).

It also includes an RSS feed. For more information about RSS feeds in general, see [aboutfeeds.com](https://aboutfeeds.com/).

## 🔨 Tools

My homepage was built using [Astro](https://docs.astro.build/en/getting-started/) and Tailwind CSS.

## 🧞 Commands

| Command          | Action                                                         |
| :--------------- | :------------------------------------------------------------- |
| `pnpm install`   | Installs dependencies                                          |
| `pnpm dev`       | Starts local dev server at `localhost:4321`                    |
| `pnpm build`     | Build your production site to `./dist/`                        |
| `pnpm postbuild` | Pagefind script to build the static search of your blog posts  |
| `pnpm preview`   | Preview your build locally, before deploying                   |
| `pnpm sync`      | Generate types based on your config in `src/content/config.ts` |

## 📝 Custom Directives

This blog supports several custom Markdown directives and syntax extensions for enhanced content:

### Admonitions

Create callout boxes with five different types: `tip`, `note`, `important`, `caution`, `warning`.

```markdown
:::note
This is a note with default title.
:::

:::tip[Custom Title Here]
This is a tip with a custom title.
:::
```

### Figure Directive

Wrap images or videos with optional captions:

```markdown
:::figure{.fullwidth}
![Alt text](image.png)

This becomes the caption. Supports **markdown** formatting.
:::
```

### GitHub Cards

Embed GitHub repository or user cards that fetch live data:

```markdown
::github{repo="username/repository"}

::github{user="username"}
```

### Quote Attribution

Automatically format blockquote attributions using `--` or `—`:

```markdown
> This is a quote that spans multiple lines.
> It continues here.
>
> -- Author Name, _Book Title_
```

### Sidenotes & Margin Notes

Tufte-style margin notes that appear in the sidebar on desktop and inline on mobile:

**Sidenotes (numbered):**

```markdown
Text with a reference[^1] to a sidenote.

[^1]: Content of the sidenote. Supports **markdown**.
```

**Margin Notes (unnumbered, with ⊕ symbol):**

```markdown
Text with a reference[^label] to a margin note.

[^label]: Content of the margin note.
```

### Book Cards

Embed a rich book card with cover art, metadata, and ratings, resolved at build time from an ISBN:

```markdown
::book{isbn="9780756404741" url="https://www.goodreads.com/book/show/..." rating=4.5 goodreads=4.52}
```

| Attribute    | Required | Description                                                            |
| :----------- | :------- | :--------------------------------------------------------------------- |
| `isbn`       | Yes      | ISBN-10 or ISBN-13 (hyphens optional)                                  |
| `url`        | No       | Link for the card; defaults to a Goodreads search for the ISBN         |
| `rating`     | No       | Your personal rating (1–5, fractional ok)                              |
| `goodreads`  | No       | Goodreads community rating (1–5, fractional ok)                        |
| `title`      | No       | Override the fetched title                                             |
| `author`     | No       | Override the fetched author                                            |
| `cover`      | No       | Override the fetched cover image URL                                   |
| `year`       | No       | Override the publication year (useful for classics)                    |
| `pages`      | No       | Override the page count                                                |

Metadata is fetched from Open Library and cached in `src/data/book-cache.json`. Cover images are downloaded, optimized to WebP, and stored in `public/book-covers/`. Delete a book's cache entry or cover file to force a refresh.

### Mathematical Notation

LaTeX-style math rendering with MathML for modern browsers:

**Inline math:**

```markdown
The equation $E = mc^2$ is famous.
```

**Block math:**

```markdown
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

## Credit

Up to date with [Astro Theme Cactus](https://github.com/chrismwilliams/astro-theme-cactus/releases) v6.11.0.
