import type { Blockquote, Paragraph, Text } from "mdast";
import type { Plugin } from "unified";
import { visit } from "unist-util-visit";

const ATTRIBUTION_PATTERN = /(?:^|\n) ?[-—]+ ?/;

/**
 * Remark plugin to convert blockquote attributions into footer elements.
 *
 * Syntax works two ways:
 *
 * With blank line:
 * > Quote text here
 * >
 * > — Author Name, *Book Title*
 *
 * Without blank line:
 * > Quote text here
 * > — Author Name, *Book Title*
 *
 * The attribution (starting with -- or —) becomes a <footer class="quote-attribution">
 */
export const remarkQuoteAttribution: Plugin = () => {
  return (tree) => {
    visit(tree, "blockquote", (node: Blockquote) => {
      // 1. Get the last child of the blockquote
      const lastParagraph = node.children[node.children.length - 1];

      // Only process if the last child is a paragraph
      if (!lastParagraph || lastParagraph.type !== "paragraph") return;

      // 2. Find the specific child node containing the attribution marker
      const childIndex = lastParagraph.children.findIndex((child) => {
        return child.type === "text" && ATTRIBUTION_PATTERN.test(child.value);
      });

      if (childIndex === -1) return;

      const textNode = lastParagraph.children[childIndex] as Text;
      const match = textNode.value.match(ATTRIBUTION_PATTERN);

      if (!match) return;

      // 3. Handle the two cases based on where the match occurred

      // CASE A: Attribution is on a new line within the text node ("\n-- Author")
      // We must split the paragraph into two: Content + Footer
      if (match[0].startsWith("\n")) {
        // Calculate split point
        const splitIndex = match.index ?? 0;

        // Extract the attribution text (removing the \n and -- part)
        const attributionText = textNode.value.slice(splitIndex + match[0].length);

        // Truncate the original text node to remove the attribution part
        textNode.value = textNode.value.slice(0, splitIndex);

        // Collect all siblings that come *after* the text node (e.g., links, italics)
        const subsequentNodes = lastParagraph.children.slice(childIndex + 1);

        // Create the new footer node
        const footerNode: Paragraph = {
          type: "paragraph",
          data: {
            hName: "footer",
            hProperties: { className: ["quote-attribution"] },
          },
          children: [
            { type: "text", value: attributionText } as Text,
            ...subsequentNodes,
          ],
        };

        // Remove moved nodes from the original paragraph
        lastParagraph.children = lastParagraph.children.slice(0, childIndex + 1);

        // Push the new footer into the blockquote
        node.children.push(footerNode);
      }

      // CASE B: Attribution is at the start of the node ("-- Author")
      // If this text node is the first child, the WHOLE paragraph is the footer.
      else if (childIndex === 0) {
        // Clean the text (remove the -- part)
        textNode.value = textNode.value.substring(match[0].length);

        // Convert the existing paragraph to a footer
        lastParagraph.data = {
          hName: "footer",
          hProperties: { className: ["quote-attribution"] },
        };
      }
    });
  };
};
