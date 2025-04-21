import FiraCodeBold from "@/assets/fira-code-700.ttf";
import FiraCode from "@/assets/fira-code-regular.ttf";
import { getAllPosts } from "@/data/post";
import { siteConfig } from "@/site.config";
import { Resvg } from "@resvg/resvg-js";
import type { APIContext, InferGetStaticPropsType } from "astro";
import satori, { type SatoriOptions } from "satori";

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
	const svg = await satori(
		{
			type: "div",
			props: {
				style: {
					background: "#fefbfb",
					width: "100%",
					height: "100%",
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
				},
				children: [
					{
						type: "div",
						props: {
							style: {
								position: "absolute",
								top: "-1px",
								right: "-1px",
								border: "4px solid #000",
								background: "#ecebeb",
								opacity: "0.9",
								borderRadius: "4px",
								display: "flex",
								justifyContent: "center",
								margin: "2.5rem",
								width: "88%",
								height: "80%",
							},
						},
					},
					{
						type: "div",
						props: {
							style: {
								border: "4px solid #000",
								background: "#fefbfb",
								borderRadius: "4px",
								display: "flex",
								justifyContent: "center",
								margin: "2rem",
								width: "88%",
								height: "80%",
							},
							children: {
								type: "div",
								props: {
									style: {
										display: "flex",
										flexDirection: "column",
										justifyContent: "space-between",
										margin: "20px",
										width: "90%",
										height: "90%",
									},
									children: [
										{
											type: "p",
											props: {
												style: {
													fontSize: 72,
													fontWeight: "bold",
													maxHeight: "84%",
													overflow: "hidden",
												},
												children: title,
											},
										},
										{
											type: "div",
											props: {
												style: {
													display: "flex",
													justifyContent: "space-between",
													width: "100%",
													marginBottom: "8px",
													fontSize: 28,
												},
												children: [
													{
														type: "span",
														props: {
															children: [
																"by ",
																{
																	type: "span",
																	props: {
																		style: { color: "transparent" },
																		children: '"',
																	},
																},
																{
																	type: "span",
																	props: {
																		style: {
																			overflow: "hidden",
																			fontWeight: "bold",
																		},
																		children: siteConfig.author,
																	},
																},
															],
														},
													},
													{
														type: "span",
														props: {
															style: { overflow: "hidden", fontWeight: "bold" },
															children: siteConfig.title,
														},
													},
												],
											},
										},
									],
								},
							},
						},
					},
				],
			},
		},
		ogOptions,
	);
	const png = new Resvg(svg).render().asPng();
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
		.filter(({ data }) => !data.ogImage)
		.map((post) => ({
			params: { slug: post.id },
			props: {
				pubDate: post.data.updatedDate ?? post.data.publishDate,
				title: post.data.title,
			},
		}));
}
