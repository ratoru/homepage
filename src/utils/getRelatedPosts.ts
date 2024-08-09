import type { CollectionEntry } from "astro:content";

/**
 * @param allPosts - the list of posts to filter
 * @param currentSlug - the slug of the current post
 * @param currentTags - the tags of the current post
 * @returns a list of at most 3 related posts
 */
export function getRelatedPosts(
  allPosts: CollectionEntry<"blog">[],
  currentSlug: string,
  currentTags: string[],
) {
  // Add tags you don't want to be recommended because they have low signal here.
  const excludedTags: string[] = [];
  const finalTags = currentTags.filter((tag) => !excludedTags.includes(tag));

  const posts = allPosts
    .filter(
      (post) =>
        post.slug != currentSlug &&
        post.data.tags?.filter((tag) => finalTags.includes(tag)).length > 0,
    )
    .map((post: CollectionEntry<"blog">) => ({
      ...post,
      sameTagCount: post.data.tags.filter((tag) => finalTags.includes(tag))
        .length,
    }))
    .sort((a, b) => {
      if (a.sameTagCount > b.sameTagCount) return -1;
      if (b.sameTagCount > a.sameTagCount) return 1;

      if (a.data.pubDate > b.data.pubDate) return -1;
      if (a.data.pubDate < b.data.pubDate) return 1;

      return 0;
    })
    .slice(0, 3);

  return posts;
}
