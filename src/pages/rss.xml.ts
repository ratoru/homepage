import { getAllPosts } from "@/data/post";
import { siteConfig } from "@/site.config";
import rss from "@astrojs/rss";

export const GET = async () => {
	const posts = await getAllPosts();

	return rss({
		title: siteConfig.title,
		description: siteConfig.description,
		site: import.meta.env.SITE,
		items: posts.map((post) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.publishDate,
			link: `posts/${post.id}/`,
		})),
		customData: [
			"<language>en-us</language>",
			`<atom:link href="${new URL("rss.xml", siteConfig.url)}" rel="self" type="application/rss+xml" />`,
		].join(""),
		stylesheet: "/rss/styles.xsl",
	});
};
