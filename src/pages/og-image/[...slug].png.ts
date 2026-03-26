import { Resvg } from "@resvg/resvg-js";
import type { APIContext, InferGetStaticPropsType } from "astro";
import satori, { type SatoriOptions } from "satori";
import FiraCodeBold from "@/assets/fira-code-700.ttf";
import FiraCode from "@/assets/fira-code-regular.ttf";
import { getAllPosts } from "@/data/post";
import { ogMarkup } from "./_ogMarkup";

const ogOptions: SatoriOptions = {
	// debug: true,
	fonts: [
		{
			data: Buffer.from(FiraCode),
			name: "Fira Code",
			style: "normal",
			weight: 400,
		},
		{
			data: Buffer.from(FiraCodeBold),
			name: "Fira Code",
			style: "normal",
			weight: 700,
		},
	],
	height: 630,
	width: 1200,
};

type Props = InferGetStaticPropsType<typeof getStaticPaths>;

export async function GET(context: APIContext) {
	const { title } = context.props as Props;

	// const postDate = getFormattedDate(pubDate, {
	//   month: "long",
	//   weekday: "long",
	// });
	const svg = await satori(ogMarkup(title), ogOptions);
	const pngBuffer = new Resvg(svg).render().asPng();
	const png = new Uint8Array(pngBuffer);
	return new Response(png, {
		headers: {
			"Cache-Control": "public, max-age=31536000, immutable",
			"Content-Type": "image/png",
		},
	});
}

export async function getStaticPaths() {
	const posts = await getAllPosts();
	return posts
		.values()
		.filter(({ data }) => !data.ogImage)
		.map((post) => ({
			params: { slug: post.id },
			props: {
				pubDate: post.data.updatedDate ?? post.data.publishDate,
				title: post.data.title,
			},
		}))
		.toArray();
}
