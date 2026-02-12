import type { Paragraph } from "mdast";
import type { ContainerDirective } from "mdast-util-directive";
import type { Plugin } from "unified";
import type { Node } from "unist";
import { visit } from "unist-util-visit";

// Helper to determine if a node is media content (not caption text)
const isMediaNode = (node: Node): boolean => {
  // 1. Direct match
  if (node.type === 'image' || node.type === 'code' || node.type === 'leafDirective') {
    return true;
  }

  // 2. Check for Image inside Paragraph (The Trap)
  if (node.type === 'paragraph') {
    const paragraph = node as Paragraph;
    // It's media if the paragraph has exactly 1 child, and that child is an image
    if (paragraph.children.length === 1 && paragraph.children[0].type === 'image') {
      return true;
    }
  }
  return false;
};

export const remarkDirectiveFigure: Plugin = () => {
	return (tree) => {
		visit(tree, (node) => {
			// Only process :::figure{} container directives
			if (
				node.type !== "containerDirective" ||
				(node as ContainerDirective).name !== "figure"
			) {
				return;
			}

			const directive = node as ContainerDirective;
			const children = directive.children || [];

			// Separate media nodes from caption nodes
			const mediaNodes: Node[] = [];
			const captionNodes: Node[] = [];

			// Track content types for CSS classes
			let hasVideo = false;
			let hasImage = false;
			let hasCode = false;

			// Determine if caption comes first (if first node is non-media)
			const captionFirst = children.length > 0 && !isMediaNode(children[0]);

			// Partition children into media and caption
			for (const child of children) {
				if (isMediaNode(child)) {
					mediaNodes.push(child);

					// Track content types
					if (child.type === "image") hasImage = true;
					if (child.type === "code") hasCode = true;
					if (
						"name" in child &&
						(child.name === "youtube" || child.name === "video")
					) {
						hasVideo = true;
					}
				} else {
					// Paragraphs and other text content become caption
					captionNodes.push(child);
				}
			}

			// Build figcaption if we have caption content
			let figcaption: Node | null = null;
			if (captionNodes.length > 0) {
				figcaption = {
					type: "figcaption",
					data: {
						hName: "figcaption",
					},
					children: captionNodes,
				} as Node;
			}

			// Rebuild children array with proper ordering
			const newChildren: Node[] = [];

			if (captionFirst && figcaption) {
				newChildren.push(figcaption);
				newChildren.push(...mediaNodes);
			} else {
				newChildren.push(...mediaNodes);
				if (figcaption) {
					newChildren.push(figcaption);
				}
			}

			// Build class list from attributes and content types
			const existingClasses =
				typeof directive.attributes?.class === "string"
					? directive.attributes.class.split(" ")
					: [];

			const contentTypeClasses = [
				hasVideo ? "video" : "",
				hasImage ? "image" : "",
				hasCode ? "code" : "",
			].filter(Boolean);

			const allClasses = [...existingClasses, ...contentTypeClasses].filter(Boolean);

			// Transform the directive into a figure element
			directive.children = newChildren as typeof directive.children;
			directive.data = {
				...directive.data,
				hName: "figure",
				hProperties: {
					...(directive.attributes || {}),
					className: allClasses,
				},
			};
		});
	};
};

export default remarkDirectiveFigure;
