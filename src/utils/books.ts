import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

/** Metadata we resolve for a book, keyed by ISBN. */
export interface BookMeta {
	title: string;
	author: string;
	/** Public path to the optimized cover, e.g. `/book-covers/9780.....webp`, or "" if none. */
	cover: string;
	/** Four-digit publication year (from the edition's publish date), or "". */
	year: string;
	/** Page count, or 0 if unknown. */
	pages: number;
}

const CACHE_PATH = path.join(process.cwd(), "src/data/book-cache.json");
const COVERS_DIR = path.join(process.cwd(), "public/book-covers");
const COVERS_PUBLIC = "/book-covers";

/**
 * Memoized as a single shared promise so concurrent first-time lookups all resolve to the
 * *same* object — otherwise each would readFile and reassign, and their writes would clobber.
 */
let cachePromise: Promise<Record<string, BookMeta>> | null = null;
/** Dedupe concurrent first-time lookups of the same ISBN across parallel page renders. */
const inFlight = new Map<string, Promise<BookMeta>>();
/** Serialize cache writes so parallel renders can't interleave and corrupt the JSON. */
let writeChain: Promise<void> = Promise.resolve();

function loadCache(): Promise<Record<string, BookMeta>> {
	if (!cachePromise) {
		cachePromise = fs
			.readFile(CACHE_PATH, "utf-8")
			.then((raw) => JSON.parse(raw) as Record<string, BookMeta>)
			.catch(() => ({}));
	}
	return cachePromise;
}

function saveCache(): Promise<void> {
	writeChain = writeChain.then(async () => {
		const c = await loadCache();
		await fs.mkdir(path.dirname(CACHE_PATH), { recursive: true });
		await fs.writeFile(CACHE_PATH, `${JSON.stringify(c, null, 2)}\n`);
	});
	return writeChain;
}

/** Strip hyphens/spaces so "978-0-7564-0474-1" matches "9780756404741". */
export function normalizeIsbn(isbn: string): string {
	return isbn.replace(/[^0-9Xx]/g, "");
}

interface FetchedMeta {
	title?: string | undefined;
	author?: string | undefined;
	coverUrl?: string | undefined;
	year?: string | undefined;
	pages?: number | undefined;
}

/**
 * Open Library's Books API returns title, author names, and cover URLs in a single
 * keyless call with no rate quota — unlike keyless Google Books, which shares one
 * global daily quota across all anonymous callers and is routinely exhausted.
 */
async function fetchOpenLibrary(isbn: string): Promise<FetchedMeta | null> {
	const res = await fetch(
		`https://openlibrary.org/api/books?bibkeys=ISBN:${isbn}&format=json&jscmd=data`,
	);
	if (!res.ok) return null;
	const data = (await res.json()) as Record<
		string,
		{
			title?: string;
			authors?: { name?: string }[];
			cover?: Record<string, string>;
			publish_date?: string;
			number_of_pages?: number;
		}
	>;
	const book = data[`ISBN:${isbn}`];
	if (!book) return null;
	return {
		title: book.title,
		// First author only: OL lists can include cover artists / translators / duplicates.
		author: book.authors?.[0]?.name,
		coverUrl: book.cover?.large ?? `https://covers.openlibrary.org/b/isbn/${isbn}-L.jpg`,
		// publish_date comes in mixed formats ("2008 April", "May 04, 2021") — pull the year.
		year: book.publish_date?.match(/\d{4}/)?.[0] ?? "",
		pages: book.number_of_pages ?? 0,
	};
}

/**
 * Download a cover, optimize to webp, and return its public path (or undefined).
 * Skips the network if the file already exists — delete it to force a redownload.
 */
async function downloadCover(isbn: string, url: string): Promise<string | undefined> {
	const out = path.join(COVERS_DIR, `${isbn}.webp`);
	const publicPath = `${COVERS_PUBLIC}/${isbn}.webp`;
	try {
		await fs.access(out);
		return publicPath; // already downloaded on a previous build
	} catch {
		// not downloaded yet
	}
	try {
		const res = await fetch(url);
		if (!res.ok) return undefined;
		const buf = Buffer.from(await res.arrayBuffer());
		if (buf.byteLength < 1000) return undefined; // Open Library returns a tiny blank for missing covers
		await fs.mkdir(COVERS_DIR, { recursive: true });
		await sharp(buf)
			.resize({ width: 400, withoutEnlargement: true })
			.webp({ quality: 80 })
			.toFile(out);
		return publicPath;
	} catch {
		return undefined;
	}
}

async function resolve(isbn: string): Promise<BookMeta> {
	const c = await loadCache();
	const meta = await fetchOpenLibrary(isbn).catch(() => null);
	const cover = meta?.coverUrl ? await downloadCover(isbn, meta.coverUrl) : undefined;
	c[isbn] = {
		title: meta?.title ?? "Unknown title",
		author: meta?.author ?? "Unknown author",
		cover: cover ?? "",
		year: meta?.year ?? "",
		pages: meta?.pages ?? 0,
	};
	await saveCache();
	return c[isbn];
}

/**
 * Resolve a book's title/author/cover from its ISBN, using a committed on-disk cache.
 * A cached ISBN is returned with no network. To refetch a book, delete its cover in
 * `public/book-covers/` and/or its entry in `book-cache.json`. `overrides` win over
 * fetched values.
 */
export async function getBookMeta(
	rawIsbn: string,
	overrides: Partial<BookMeta> = {},
): Promise<BookMeta> {
	const isbn = normalizeIsbn(rawIsbn);
	const c = await loadCache();

	// `pages === undefined` also catches entries cached before pages/year were added.
	let base = c[isbn];
	if (!base || base.pages === undefined) {
		let pending = inFlight.get(isbn);
		if (!pending) {
			pending = resolve(isbn);
			inFlight.set(isbn, pending);
			pending.finally(() => inFlight.delete(isbn));
		}
		base = await pending;
	}

	return {
		title: overrides.title ?? base.title,
		author: overrides.author ?? base.author,
		cover: overrides.cover ?? base.cover,
		year: overrides.year ?? base.year,
		pages: overrides.pages ?? base.pages,
	};
}
