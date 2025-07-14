import type { AstroExpressiveCodeOptions } from "astro-expressive-code";
import type { SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	// Used as both a meta property (src/components/BaseHead.astro L:31 + L:49) & the generated satori png (src/pages/og-image/[slug].png.ts)
	author: "Raphael Ruban",
	// Date.prototype.toLocaleDateString() parameters, found in src/utils/date.ts.
	date: {
		locale: "en-GB",
		options: {
			day: "numeric",
			month: "short",
			year: "numeric",
		},
	},
	// Used as the default description meta property and webmanifest description
	description: "Writings about Engineering and Health.",
	// HTML lang property, found in src/layouts/Base.astro L:18 & astro.config.ts L:48
	lang: "en-US",
	// Meta property, found in src/components/BaseHead.astro L:42
	ogLocale: "en_US",
	// Used to construct the meta title property found in src/components/BaseHead.astro L:11, and webmanifest name found in astro.config.ts L:42
	title: "Raphael's Rabbit Holes",
	// ! Please remember to replace the following site property with your own domain, used in astro.config.ts
	url: "https://ratoru.com/",
};

// Used to generate links in both the Header & Footer.
export const menuLinks: { path: string; title: string }[] = [
	{
		path: "/blog/",
		title: "Blog",
	},
	{
		path: "/posts/",
		title: "All Posts",
	},
	{
		path: "/tags/",
		title: "Tags",
	},
];

/**
  Uses https://www.astroicon.dev/getting-started/
  Find icons via guide: https://www.astroicon.dev/guides/customization/#open-source-icon-sets
  Only installed pack is: @iconify-json/mdi
*/
export const socialLinks: {
	friendlyName: string;
	link: string;
	name: string;
}[] = [
	{
		friendlyName: "Github",
		link: "https://github.com/ratoru/",
		name: "tabler:brand-github",
	},
	{
		friendlyName: "LinkedIn",
		link: "https://linkedin.com/in/ratoru",
		name: "tabler:brand-linkedin",
	},
	{
		friendlyName: "Homepage",
		link: "https://ratoru.com/",
		name: "tabler:home-link",
	},
];

// https://expressive-code.com/reference/configuration/
export const expressiveCodeOptions: AstroExpressiveCodeOptions = {
	styleOverrides: {
		borderRadius: "4px",
		codeFontFamily:
			'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
		codeFontSize: "0.875rem",
		codeLineHeight: "1.7142857rem",
		codePaddingInline: "1rem",
		frames: {
			frameBoxShadowCssValue: "none",
		},
		uiLineHeight: "inherit",
	},
	themeCssSelector(theme, { styleVariants }) {
		// If one dark and one light theme are available
		// generate theme CSS selectors compatible with cactus-theme dark mode switch
		if (styleVariants.length >= 2) {
			const baseTheme = styleVariants[0]?.theme;
			const altTheme = styleVariants.find((v) => v.theme.type !== baseTheme?.type)?.theme;
			if (theme === baseTheme || theme === altTheme) return `[data-theme='${theme.type}']`;
		}
		// return default selector
		return `[data-theme="${theme.name}"]`;
	},
	// One dark, one light theme => https://expressive-code.com/guides/themes/#available-themes
	themes: ["dracula", "github-light"],
	useThemedScrollbars: false,
};
