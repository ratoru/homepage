import type { Paragraph, Root } from "mdast";
import type { Plugin } from "unified";
import { visit } from "unist-util-visit";
import { type BookMeta, getBookMeta } from "../utils/books";
import { h, isNodeDirective } from "../utils/remark";

const DIRECTIVE_NAME = "book";

/** Goodreads' worded scale, indexed 1–5. */
const RATING_WORDS = [
	"",
	"did not like it",
	"it was ok",
	"liked it",
	"really liked it",
	"it was amazing",
];

/** Word for a (possibly fractional) rating, biased to the achieved level (4.5 → "really liked it"). */
function ratingWord(rating: number): string {
	const i = Math.min(5, Math.max(1, Math.floor(rating + 0.25)));
	return RATING_WORDS[i]!;
}

/** Render a fractional star row driven by the `--rating` custom property in CSS. */
function stars(rating: number, extraClass: string, ariaLabel: string, tip: string): Paragraph {
	return h(
		"span",
		{
			class: `stars ${extraClass}`.trim(),
			style: `--rating: ${rating}`,
			role: "img",
			"aria-label": ariaLabel,
			"data-tip": tip,
		},
		[],
	);
}

/** One rating row: [label] [stars] [value], with a Goodreads-style tooltip on hover. */
function ratingRow(
	rating: number,
	opts: { label: string; modifier: string; small?: boolean; tip: string },
): Paragraph {
	return h("span", { class: `book-rating ${opts.modifier}` }, [
		stars(rating, opts.small ? "stars--sm" : "", `${opts.label}: ${rating} out of 5`, opts.tip),
		h("span", { class: "book-rating-value" }, [{ type: "text", value: String(rating) }]),
		h("span", { class: "book-rating-label" }, [{ type: "text", value: opts.label }]),
	]);
}

interface CardOptions {
	meta: BookMeta;
	rating?: number | undefined;
	goodreads?: number | undefined;
	url: string;
}

function buildCard({ meta, rating, goodreads, url }: CardOptions): Paragraph {
	const ratings: Paragraph[] = [];
	if (rating !== undefined && !Number.isNaN(rating)) {
		ratings.push(
			ratingRow(rating, {
				label: "My Rating",
				modifier: "book-rating--mine",
				tip: ratingWord(rating),
			}),
		);
	}
	if (goodreads !== undefined && !Number.isNaN(goodreads)) {
		ratings.push(
			ratingRow(goodreads, {
				label: "Goodreads",
				modifier: "book-rating--goodreads",
				small: true,
				tip: ratingWord(goodreads),
			}),
		);
	}

	const head: Paragraph[] = [
		h("div", { class: "book-titlerow" }, [
			h("a", { class: "book-title", href: url }, [{ type: "text", value: meta.title }]),
			h("a", { class: "book-link", href: url, "aria-label": `${meta.title} on Goodreads` }, [
				h("span", { class: "book-link-icon" }, []),
			]),
		]),
		h("p", { class: "book-author" }, [{ type: "text", value: meta.author }]),
	];

	const metaParts: string[] = [];
	if (meta.year) metaParts.push(meta.year);
	if (meta.pages) metaParts.push(`${meta.pages} pages`);
	if (metaParts.length) {
		head.push(h("p", { class: "book-meta" }, [{ type: "text", value: metaParts.join(" · ") }]));
	}

	const body = h("div", { class: "book-body" }, [
		h("div", { class: "book-head" }, head),
		h("div", { class: "book-ratings" }, ratings),
	]);

	const children: Paragraph[] = [];
	if (meta.cover) {
		children.push(
			h("a", { class: "book-cover", href: url, "aria-hidden": "true", tabindex: "-1" }, [
				h("img", { src: meta.cover, alt: `Cover of ${meta.title}`, loading: "lazy" }, []),
			]),
		);
	}
	children.push(body);

	return h("aside", { class: "book-card not-prose" }, children);
}

/**
 * Turns `::book{isbn="9780756404741" rating=4.5 goodreads=4.52 url="..."}` into a book
 * card. Title/author/cover/year/pages are resolved from the ISBN at build time (cached);
 * the ratings and link are authored locally. Optional `title`/`author`/`cover`/`year`/
 * `pages` attributes override the fetched metadata (e.g. `year` for a classic's original
 * publication year).
 */
export const remarkBookCard: Plugin<[], Root> = () => async (tree) => {
	const jobs: Promise<void>[] = [];

	visit(tree, (node, index, parent) => {
		if (!parent || index === undefined || !isNodeDirective(node)) return;
		if (node.type !== "leafDirective" || node.name !== DIRECTIVE_NAME) return;

		const attrs = node.attributes ?? {};
		const isbn = attrs.isbn;
		if (!isbn) return; // Leave the directive as-is if no ISBN is provided

		const rating = attrs.rating ? Number(attrs.rating) : undefined;
		const goodreads = attrs.goodreads ? Number(attrs.goodreads) : undefined;
		const url = attrs.url ?? `https://www.goodreads.com/search?q=${encodeURIComponent(isbn)}`;
		const overrides: Partial<BookMeta> = {};
		if (attrs.title) overrides.title = attrs.title;
		if (attrs.author) overrides.author = attrs.author;
		if (attrs.cover) overrides.cover = attrs.cover;
		if (attrs.year) overrides.year = attrs.year;
		if (attrs.pages) overrides.pages = Number(attrs.pages);

		const targetParent = parent;
		const targetIndex = index;

		jobs.push(
			getBookMeta(isbn, overrides).then((meta) => {
				// 1:1 replacement keeps sibling indices stable across concurrent jobs.
				targetParent.children[targetIndex] = buildCard({ meta, rating, goodreads, url });
			}),
		);
	});

	await Promise.all(jobs);
};
