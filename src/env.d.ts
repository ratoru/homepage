declare module "@pagefind/default-ui" {
	declare class PagefindUI {
		constructor(arg: unknown);
	}
}

declare module "@fontsource-variable/alegreya" {}

interface Window {
	umami?: { track: (event?: string, data?: Record<string, unknown>) => void };
}
