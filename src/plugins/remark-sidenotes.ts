import { createHash } from "node:crypto";
import type {
	BlockContent,
	DefinitionContent,
	FootnoteDefinition,
	FootnoteReference,
	Parent,
	PhrasingContent,
	Root,
} from "mdast";
import type { Plugin } from "unified";
import { u } from "unist-builder";
import { visit } from "unist-util-visit";

// Configuration interface
interface SidenoteOptions {
	marginnoteLabel?: string;
}

// Custom Node Types for TypeScript
interface SidenoteNode extends Parent {
	type: "sidenote";
	data: {
		hName: string;
		hProperties: Record<string, unknown>;
	};
	children: Array<BlockContent | DefinitionContent | PhrasingContent>;
}

interface FootnoteNode extends Parent {
	type: "footnote";
	children: PhrasingContent[];
}

const DEFAULT_LABEL = "\u2295"; // ⊕

/**
 * Helper to determine if a string represents a number.
 * Used to distinguish numbered Sidenotes from symbol-based Marginnotes.
 */
const isNumeric = (val: string): boolean => !Number.isNaN(Number(val));

/**
 * Generates a stable hash for inline notes to serve as an ID.
 */
const generateContentHash = (content: string): string => {
	return createHash("sha1").update(content).digest("hex").slice(0, 7);
};

/**
 * Factory function to build the HAST-compatible AST structure.
 * Structure: <span class="wrapper"> <label> <input> <span class="content"> </span>
 */
const createSidenoteNode = (
	id: string,
	isMarginNote: boolean,
	children: Array<BlockContent | DefinitionContent | PhrasingContent>,
	labelSymbol: string,
): SidenoteNode => {
	const typeClass = isMarginNote ? "marginnote" : "sidenote";
	const inputId = `${typeClass}-${id}`;

	// 1. The Label (Clickable toggle)
	const labelChildren = isMarginNote
		? [u("text", labelSymbol)]
		: [
				u(
					"element",
					{
						data: {
							hName: "span",
							hProperties: { className: ["sr-only"] },
						},
					},
					[u("text", "Sidenote")],
				),
			];

	const labelNode = u(
		"element",
		{
			data: {
				hName: "label",
				hProperties: {
					htmlFor: inputId,
					className: ["margin-toggle", isMarginNote ? "" : "sidenote-number"].filter(Boolean),
				},
			},
		},
		labelChildren,
	);

	// 2. The Checkbox (Hidden state controller)
	const inputNode = u(
		"element",
		{
			data: {
				hName: "input",
				hProperties: {
					type: "checkbox",
					id: inputId,
					className: ["margin-toggle"],
				},
			},
		},
		[],
	);

	// 3. The Content
	const contentNode = u(
		"element",
		{
			data: {
				hName: "span",
				hProperties: {
					className: [`${typeClass}-definition`],
				},
			},
		},
		children,
	);

	// Wrapper Span
	return u(
		"sidenote",
		{
			data: {
				hName: "span",
				hProperties: { className: [typeClass] },
			},
		},
		[labelNode as never, inputNode as never, contentNode as never],
	) as SidenoteNode;
};

/**
 * Remark plugin to convert footnotes into sidenotes and margin notes
 *
 * Syntax:
 * - Sidenotes (numbered): [^1] with definition [^1]: content
 * - Margin notes (unnumbered): [^label] with definition [^label]: content
 * - Inline margin notes: ^[content]
 *
 * Features:
 * - Responsive: inline on mobile, margin on desktop
 * - CSS-only toggle mechanism (no JS required)
 * - Automatic numbering via CSS counters
 */
const remarkSidenotes: Plugin<[SidenoteOptions?], Root> = (options = {}) => {
	const marginSymbol = options.marginnoteLabel || DEFAULT_LABEL;

	return (tree: Root) => {
		const definitions = new Map<string, FootnoteDefinition>();
		const definitionsToRemove = new Set<FootnoteDefinition>();

		// Pass 1: Collect and index all definitions
		visit(tree, "footnoteDefinition", (node) => {
			definitions.set(node.identifier, node);
			definitionsToRemove.add(node);
		});

		// Pass 2: Transform references and inline notes
		visit(tree, ["footnoteReference", "footnote"], (node, index, parent) => {
			if (!parent || index === undefined) return;

			let children: Array<BlockContent | DefinitionContent | PhrasingContent> = [];
			let identifier = "";
			let isMarginNote = false;

			// Handle standard references [^1] or [^label]
			if (node.type === "footnoteReference") {
				const refNode = node as FootnoteReference;
				identifier = refNode.identifier;
				const def = definitions.get(identifier);

				// If no definition found, leave the reference as-is (graceful degradation)
				if (!def) return;

				// Unwrap paragraph if it's the only child (prevents <p> inside <span>)
				children =
					def.children.length === 1 && def.children[0].type === "paragraph"
						? def.children[0].children
						: def.children;

				isMarginNote = !isNumeric(identifier);
			}
			// Handle inline notes ^[text]
			else if (node.type === "footnote") {
				const inlineNode = node as FootnoteNode;
				children = inlineNode.children;
				// Generate an ID based on content to link label/input
				const textContent = children
					.map((c) => {
						if ("value" in c) return c.value;
						return "";
					})
					.join("");
				identifier = generateContentHash(textContent);
				isMarginNote = true; // Inline notes are always treated as margin notes (unnumbered)
			}

			// Create replacement node
			const replacement = createSidenoteNode(identifier, isMarginNote, children, marginSymbol);

			// Replace the original node
			parent.children.splice(index, 1, replacement);
		});

		// Pass 3: Cleanup definitions
		// We filter the tree to remove definitions we used.
		// This is safer than splicing during the first visit.
		const removeDefinitions = (node: Parent) => {
			node.children = node.children.filter((child) => {
				if (child.type === "footnoteDefinition") {
					return !definitionsToRemove.has(child as FootnoteDefinition);
				}
				if ("children" in child) {
					removeDefinitions(child as Parent);
				}
				return true;
			});
		};

		removeDefinitions(tree);
	};
};

export default remarkSidenotes;
