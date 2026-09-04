import { describe, expect, it } from "vitest";
import { createWatchToolDefinition, type WatchOperations } from "../src/core/tools/watch.ts";

/**
 * The network is stubbed; the SHAPE is not.
 *
 * What is worth pinning here is not that Gemini can describe a video - that is
 * measured against real clips, not in a unit test. It is the two properties the
 * tool exists for and which a refactor could silently drop: the caller's own
 * question reaches the model instead of the built-in one, and the stills come
 * back in the same result as the prose. A watch that quietly returns only words
 * is the failure this tool was written to prevent, and it looks like success.
 */

const ONE_PIXEL_JPEG = Buffer.from(
	"/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==",
	"base64",
);

function stubOps(overrides: Partial<WatchOperations> = {}): WatchOperations {
	return {
		readFile: async (p) => (p.endsWith(".jpg") ? ONE_PIXEL_JPEG : Buffer.from("fake-video-bytes")),
		exec: async (command) =>
			command.startsWith("ls ")
				? { exitCode: 0, stdout: "/tmp/d/f01.jpg\n/tmp/d/f02.jpg\n", stderr: "" }
				: { exitCode: 0, stdout: "", stderr: "" },
		...overrides,
	};
}

/** Stand in for the Files API + generateContent, and record what was asked. */
function stubGemini(answer = "it is a video of a cat") {
	const seen: { uploads: number; questions: string[]; models: string[] } = { uploads: 0, questions: [], models: [] };
	const original = globalThis.fetch;
	globalThis.fetch = (async (input: any, init?: any) => {
		const url = String(input);
		if (url.includes("/upload/v1beta/files")) {
			seen.uploads++;
			return new Response(JSON.stringify({ file: { name: "files/x", uri: "https://f/x", state: "ACTIVE" } }), {
				status: 200,
			});
		}
		if (url.includes(":generateContent")) {
			seen.models.push(url.split("/models/")[1].split(":")[0]);
			const body = JSON.parse(init.body);
			seen.questions.push(body.contents[0].parts.at(-1).text);
			return new Response(JSON.stringify({ candidates: [{ content: { parts: [{ text: answer }] } }] }), {
				status: 200,
			});
		}
		throw new Error(`unexpected fetch: ${url}`);
	}) as typeof fetch;
	const restore = () => {
		globalThis.fetch = original;
	};
	return { seen, restore };
}

describe("watch tool", () => {
	it("asks the caller's question, not the built-in one", async () => {
		const g = stubGemini();
		try {
			const tool = createWatchToolDefinition("/tmp", { operations: stubOps(), apiKey: "k" });
			await tool.execute(
				"t",
				{ path: "clip.mp4", question: "did the end card land?" },
				undefined,
				undefined,
				{} as any,
			);
			expect(g.seen.questions).toEqual(["did the end card land?"]);
		} finally {
			g.restore();
		}
	});

	it("falls back to the general breakdown when no question is given", async () => {
		const g = stubGemini();
		try {
			const tool = createWatchToolDefinition("/tmp", { operations: stubOps(), apiKey: "k" });
			await tool.execute("t", { path: "clip.mp4" }, undefined, undefined, {} as any);
			expect(g.seen.questions[0]).toContain('"shots"');
			expect(g.seen.questions[0]).toContain("SOUND ON");
		} finally {
			g.restore();
		}
	});

	it("returns the frames alongside the prose", async () => {
		const g = stubGemini("a description");
		try {
			const tool = createWatchToolDefinition("/tmp", { operations: stubOps(), apiKey: "k" });
			const r = await tool.execute("t", { path: "clip.mp4", frames: 2 }, undefined, undefined, {} as any);
			const images = r.content.filter((c) => c.type === "image");
			expect(images).toHaveLength(2);
			expect(images[0]).toMatchObject({ mimeType: "image/jpeg" });
			expect(r.content.find((c) => c.type === "text")?.text).toContain("a description");
		} finally {
			g.restore();
		}
	});

	it("still answers when frames cannot be cut", async () => {
		// A box with no ffmpeg must not turn a watch into a failure. Losing the
		// evidence is bad; losing the answer as well is worse.
		const g = stubGemini("still described it");
		try {
			const tool = createWatchToolDefinition("/tmp", {
				operations: stubOps({ exec: async () => ({ exitCode: 127, stdout: "", stderr: "ffmpeg: not found" }) }),
				apiKey: "k",
			});
			const r = await tool.execute("t", { path: "clip.mp4" }, undefined, undefined, {} as any);
			expect(r.content.filter((c) => c.type === "image")).toHaveLength(0);
			const text = r.content.find((c) => c.type === "text")?.text ?? "";
			expect(text).toContain("still described it");
			expect(text).toContain("No stills");
		} finally {
			g.restore();
		}
	});

	it("refuses without a key rather than registering a tool that half works", async () => {
		const previous = process.env.GEMINI_API_KEY;
		delete process.env.GEMINI_API_KEY;
		try {
			const tool = createWatchToolDefinition("/tmp", { operations: stubOps() });
			await expect(tool.execute("t", { path: "clip.mp4" }, undefined, undefined, {} as any)).rejects.toThrow(
				/GEMINI_API_KEY/,
			);
		} finally {
			if (previous !== undefined) process.env.GEMINI_API_KEY = previous;
		}
	});

	it("refuses an empty file instead of uploading nothing", async () => {
		const g = stubGemini();
		try {
			const tool = createWatchToolDefinition("/tmp", {
				operations: stubOps({ readFile: async () => Buffer.alloc(0) }),
				apiKey: "k",
			});
			await expect(tool.execute("t", { path: "clip.mp4" }, undefined, undefined, {} as any)).rejects.toThrow(
				/nothing to watch/,
			);
			expect(g.seen.uploads).toBe(0);
		} finally {
			g.restore();
		}
	});

	it("reads the type through a signed URL's query string", async () => {
		// R2 hands out `...mov?X-Amz-Signature=...`, which used to fall through to
		// the video/mp4 default and mislabel the upload.
		const g = stubGemini();
		const original = globalThis.fetch;
		try {
			const tool = createWatchToolDefinition("/tmp", { operations: stubOps(), apiKey: "k" });
			const wrapped = globalThis.fetch;
			const types: string[] = [];
			globalThis.fetch = (async (input: any, init?: any) => {
				const url = String(input);
				if (url.startsWith("https://cdn.example")) {
					return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
				}
				if (url.includes(":generateContent")) {
					types.push(JSON.parse(init.body).contents[0].parts[0].file_data.mime_type);
				}
				return wrapped(input, init);
			}) as typeof fetch;
			await tool.execute(
				"t",
				{ path: "https://cdn.example/a/take.mov?X-Amz-Signature=abc&x=1", frames: 0 },
				undefined,
				undefined,
				{} as any,
			);
			expect(types).toEqual(["video/quicktime"]);
		} finally {
			globalThis.fetch = original;
			g.restore();
		}
	});

	it("honours a model override", async () => {
		const g = stubGemini();
		try {
			const tool = createWatchToolDefinition("/tmp", { operations: stubOps(), apiKey: "k" });
			await tool.execute("t", { path: "clip.mp4", model: "gemini-3.5-flash" }, undefined, undefined, {} as any);
			expect(g.seen.models).toEqual(["gemini-3.5-flash"]);
		} finally {
			g.restore();
		}
	});
});
