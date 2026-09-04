import { exec as nodeExec } from "node:child_process";
import { readFile as fsReadFile } from "node:fs/promises";
import { promisify } from "node:util";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { Text } from "@earendil-works/pi-tui";
import { type Static, Type } from "typebox";
import type { ToolDefinition, ToolRenderResultOptions } from "../extensions/types.ts";
import { resolveReadPath } from "./path-utils.ts";
import { getTextOutput, invalidArgText, shortenPath, str } from "./render-utils.ts";
import { wrapToolDefinition } from "./tool-definition-wrapper.ts";

/**
 * watch — put eyes and ears on a file the model cannot open itself.
 *
 * WHY THIS IS A TOOL AND NOT A SCRIPT. The capability already existed four times
 * over on the laptop (video_judge, the decompiler, screenplay/watch.py, the
 * caption passes). None of it reaches the agent: those all live under `scripts/`
 * and in host-side packages, and the sandbox receives neither. So an agent that
 * had just rendered a film could not look at it. That, not the vision gap, is
 * what this closes.
 *
 * IT ANSWERS THE CALLER'S QUESTION, NOT A FIXED ONE. Every existing driver bakes
 * its own schema — thirteen fields about an ad, six rubric axes about a render.
 * A fixed schema decides what counts, and the failure is quiet: ask "did the end
 * card land", get back a `why_it_holds` about the hook, read it as the answer.
 * So `question` is a parameter. Omit it and the default breakdown runs, because
 * that is what is wanted most of the time.
 *
 * IT RETURNS FRAMES AS WELL AS PROSE. Prose is a second model's conclusion about
 * your work. Handing back only the conclusion is how an agent comes to trust a
 * summary it cannot check — a mistake this codebase has already paid for once.
 * The stills ride along in the same result so the caller can disagree.
 */

const DEFAULT_QUESTION = `Watch this with the SOUND ON and describe what is actually there. Return JSON only:

{
 "summary": "what happens, in two or three sentences",
 "spoken": "every spoken word, verbatim. empty string if nothing is said",
 "audio": "music, sfx, room tone. is any voice studio-clean or roomy?",
 "hook_first_2s": "exactly what happens visually AND audibly in the first 2 seconds",
 "shots": [{"t_start":0.0,"t_end":0.0,"framing":"wide/mid/CU/macro","camera":"locked/push/handheld/whip/orbit - and does the CAMERA move or does the SUBJECT move?","what_happens":""}],
 "cut_count": 0,
 "onscreen_text": "any captions or graphics, verbatim, and their style",
 "tells": "anything that reads as AI-generated, broken, or unfinished - be specific and critical"
}`;

const FRAME_DEFAULT = 6;
const FRAME_MAX = 12;
/** Files API cap. A longer clip must be trimmed before it can be watched. */
const MAX_BYTES = 2 * 1024 * 1024 * 1024;
const UPLOAD_HOST = "https://generativelanguage.googleapis.com";
/** How long to wait for the Files API to finish PROCESSING a video. */
const ACTIVE_TIMEOUT_MS = 180_000;

const watchSchema = Type.Object({
	path: Type.String({
		description: "The video or audio file to watch. A path (relative or absolute), or an http(s) URL.",
	}),
	question: Type.Optional(
		Type.String({
			description:
				"What you want to know about it, in your own words. Omit for a general breakdown (shots, cuts, speech, hook, on-screen text, tells).",
		}),
	),
	frames: Type.Optional(
		Type.Number({
			description: `How many evenly-spaced stills to return alongside the answer, so you can check it yourself. Default ${FRAME_DEFAULT}, max ${FRAME_MAX}, 0 for none.`,
		}),
	),
	model: Type.Optional(Type.String({ description: "Override the Gemini model that watches." })),
});

export type WatchToolInput = Static<typeof watchSchema>;

export interface WatchToolDetails {
	model?: string;
	seconds?: number;
	frameCount?: number;
	sourceBytes?: number;
}

export interface ExecOutcome {
	exitCode: number | null;
	stdout: string;
	stderr: string;
}

/**
 * Pluggable operations, for the same reason read and bash have them: the agent
 * loop runs host-side while the file it wants to watch lives in a sandbox.
 * `exec` is here because frames are cut with ffmpeg, which must run wherever the
 * file is — a host-side ffmpeg cannot see a path inside the box.
 */
export interface WatchOperations {
	readFile: (absolutePath: string) => Promise<Buffer>;
	exec: (command: string, signal?: AbortSignal) => Promise<ExecOutcome>;
}

const execFileAsync = promisify(nodeExec);

const defaultWatchOperations: WatchOperations = {
	readFile: (path) => fsReadFile(path),
	async exec(command, signal) {
		try {
			const { stdout, stderr } = await execFileAsync(command, { signal, maxBuffer: 32 * 1024 * 1024 });
			return { exitCode: 0, stdout, stderr };
		} catch (e: any) {
			return {
				exitCode: typeof e?.code === "number" ? e.code : 1,
				stdout: e?.stdout ?? "",
				stderr: e?.stderr ?? String(e),
			};
		}
	},
};

export interface WatchToolOptions {
	/** Where the file lives and where ffmpeg runs. Default: this machine. */
	operations?: WatchOperations;
	/** Default: GEMINI_API_KEY from the environment. */
	apiKey?: string;
	/** Default: gemini-3.8-flash. */
	model?: string;
	/** A writable directory, on the same machine as `operations`, for cut frames. */
	tmpDir?: string;
}

const DEFAULT_MODEL = "gemini-3.8-flash";

function isUrl(path: string): boolean {
	return /^https?:\/\//i.test(path);
}

function mimeFor(path: string): string {
	// Strip a query string and fragment before looking at the extension. Every
	// signed URL in this codebase ends `...mp4?X-Amz-Signature=...`, and reading
	// the extension off the raw string yields "mp4?x-amz-signature=..." - no table
	// hit, and the clip gets uploaded as the fallback type. R2 is the normal way a
	// rendered film is handed around here, so this is the common case, not the edge.
	const clean = path.split("?")[0].split("#")[0];
	const ext = clean.split(".").pop()?.toLowerCase() ?? "";
	const table: Record<string, string> = {
		mp4: "video/mp4",
		mov: "video/quicktime",
		webm: "video/webm",
		mkv: "video/x-matroska",
		avi: "video/x-msvideo",
		mp3: "audio/mp3",
		wav: "audio/wav",
		m4a: "audio/mp4",
		aac: "audio/aac",
		ogg: "audio/ogg",
		flac: "audio/flac",
	};
	return table[ext] ?? "video/mp4";
}

/** Upload, then wait for the file to leave PROCESSING. A video is not readable the instant it lands. */
async function uploadAndActivate(
	bytes: Buffer,
	mimeType: string,
	apiKey: string,
	signal?: AbortSignal,
): Promise<string> {
	const res = await fetch(`${UPLOAD_HOST}/upload/v1beta/files?key=${apiKey}`, {
		method: "POST",
		headers: {
			"X-Goog-Upload-Protocol": "raw",
			"X-Goog-Upload-Content-Length": String(bytes.length),
			"X-Goog-Upload-Header-Content-Type": mimeType,
			"Content-Type": mimeType,
		},
		body: new Uint8Array(bytes),
		signal,
	});
	if (!res.ok) {
		throw new Error(`upload failed (${res.status}): ${(await res.text()).slice(0, 400)}`);
	}
	const uploaded = (await res.json()) as { file?: { name?: string; uri?: string; state?: string } };
	const name = uploaded.file?.name;
	const uri = uploaded.file?.uri;
	if (!name || !uri) throw new Error(`upload returned no file handle: ${JSON.stringify(uploaded).slice(0, 300)}`);

	let state = uploaded.file?.state ?? "PROCESSING";
	const deadline = Date.now() + ACTIVE_TIMEOUT_MS;
	while (state === "PROCESSING") {
		if (Date.now() > deadline) throw new Error(`the file was still PROCESSING after ${ACTIVE_TIMEOUT_MS / 1000}s`);
		await new Promise((r) => setTimeout(r, 1500));
		if (signal?.aborted) throw new Error("Operation aborted");
		const poll = await fetch(`${UPLOAD_HOST}/v1beta/${name}?key=${apiKey}`, { signal });
		if (!poll.ok) throw new Error(`file poll failed (${poll.status})`);
		state = ((await poll.json()) as { state?: string }).state ?? "PROCESSING";
	}
	if (state !== "ACTIVE") throw new Error(`the file ended in state ${state}, not ACTIVE`);
	return uri;
}

async function askGemini(
	fileUri: string,
	mimeType: string,
	question: string,
	model: string,
	apiKey: string,
	signal?: AbortSignal,
): Promise<string> {
	const res = await fetch(`${UPLOAD_HOST}/v1beta/models/${model}:generateContent?key=${apiKey}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			contents: [{ parts: [{ file_data: { mime_type: mimeType, file_uri: fileUri } }, { text: question }] }],
		}),
		signal,
	});
	if (!res.ok) {
		throw new Error(`generateContent failed (${res.status}): ${(await res.text()).slice(0, 400)}`);
	}
	const body = (await res.json()) as {
		candidates?: { content?: { parts?: { text?: string }[] }; finishReason?: string }[];
	};
	const parts = body.candidates?.[0]?.content?.parts ?? [];
	const text = parts
		.map((p) => p.text ?? "")
		.join("")
		.trim();
	if (!text) {
		const reason = body.candidates?.[0]?.finishReason;
		throw new Error(`the model returned no text${reason ? ` (finishReason ${reason})` : ""}`);
	}
	return text;
}

/**
 * Cut evenly-spaced stills, on the machine the file lives on.
 *
 * Best-effort by design: a clip that cannot be sampled (no ffmpeg in the box, an
 * audio-only file) must still come back with its answer. A missing picture is a
 * smaller loss than a failed watch.
 */
async function cutFrames(
	ops: WatchOperations,
	sourcePath: string,
	count: number,
	tmpDir: string,
	signal?: AbortSignal,
): Promise<{ frames: { data: string; mimeType: string }[]; note?: string }> {
	if (count <= 0) return { frames: [] };
	const dir = `${tmpDir}/pi-watch-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
	const quoted = `'${sourcePath.replace(/'/g, `'\\''`)}'`;
	// `-vf thumbnail` alone picks representative frames but clusters them; the
	// fps filter spreads the sample across the whole clip, which is what makes a
	// strip worth looking at. Falls back to whatever ffmpeg managed.
	const cmd =
		`mkdir -p ${dir} && ffprobe -v error -show_entries format=duration -of csv=p=0 ${quoted} > ${dir}/dur 2>/dev/null; ` +
		`D=$(cat ${dir}/dur 2>/dev/null || echo 0); ` +
		`ffmpeg -nostdin -v error -i ${quoted} -vf "fps=${count}/\${D:-10},scale=480:-2" -frames:v ${count} -q:v 4 ${dir}/f%02d.jpg`;
	const run = await ops.exec(cmd, signal);
	let names: string[] = [];
	const listed = await ops.exec(`ls ${dir}/f*.jpg 2>/dev/null`, signal);
	if (listed.stdout.trim()) names = listed.stdout.trim().split("\n").filter(Boolean);
	const frames: { data: string; mimeType: string }[] = [];
	for (const name of names.slice(0, count)) {
		try {
			const buf = await ops.readFile(name.trim());
			frames.push({ data: buf.toString("base64"), mimeType: "image/jpeg" });
		} catch {
			// A frame that cannot be read back is skipped, not fatal.
		}
	}
	await ops.exec(`rm -rf ${dir}`, signal).catch(() => undefined);
	if (frames.length === 0) {
		const why = (run.stderr || listed.stderr || "").trim().split("\n").slice(-1)[0];
		return { frames: [], note: `No stills: ffmpeg produced none${why ? ` (${why})` : ""}.` };
	}
	return { frames };
}

function formatWatchCall(
	args: { path?: string; question?: string } | undefined,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
): string {
	const rawPath = str(args?.path);
	const path = rawPath !== null ? shortenPath(rawPath) : null;
	let text = `${theme.fg("toolTitle", theme.bold("watch"))} ${
		path === null ? invalidArgText(theme) : theme.fg("accent", path)
	}`;
	const question = str(args?.question);
	if (question) {
		const oneLine = question.replace(/\s+/g, " ");
		text += theme.fg("toolOutput", ` — ${oneLine.length > 60 ? `${oneLine.slice(0, 57)}...` : oneLine}`);
	}
	return text;
}

function formatWatchResult(
	result: {
		content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
		details?: WatchToolDetails;
	},
	options: ToolRenderResultOptions,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
	showImages: boolean,
): string {
	const output = getTextOutput(result, showImages).trim();
	let text = "";
	if (output) {
		const lines = output.split("\n");
		const maxLines = options.expanded ? lines.length : 20;
		text += `\n${lines
			.slice(0, maxLines)
			.map((line) => theme.fg("toolOutput", line))
			.join("\n")}`;
		const remaining = lines.length - maxLines;
		if (remaining > 0) text += theme.fg("muted", `\n... (${remaining} more lines)`);
	}
	const d = result.details;
	if (d?.seconds !== undefined) {
		text += `\n${theme.fg("muted", `[${d.model ?? "?"} · ${d.seconds.toFixed(1)}s · ${d.frameCount ?? 0} frames]`)}`;
	}
	return text;
}

export function createWatchToolDefinition(
	cwd: string,
	options?: WatchToolOptions,
): ToolDefinition<typeof watchSchema, WatchToolDetails | undefined> {
	const ops = options?.operations ?? defaultWatchOperations;
	const defaultModel = options?.model ?? DEFAULT_MODEL;
	const tmpDir = options?.tmpDir ?? "/tmp";
	return {
		name: "watch",
		label: "watch",
		description:
			"Watch a video (with its sound) or listen to an audio file, and get back a description in words plus evenly-spaced stills. " +
			"Use it on anything you cannot open yourself: a clip you just rendered, a competitor's ad, a screen recording, a voiceover. " +
			"Pass `question` to ask something specific — the default is a general breakdown of shots, cuts, speech, hook, on-screen text and tells. " +
			"The stills come back so you can check the description rather than take it on trust.",
		promptSnippet: "Watch a video or listen to audio",
		promptGuidelines: [
			"Use watch to look at a video you made or found. Do not reason about a clip from its filename, its duration, or the command that produced it.",
			"The words in a watch result are another model's opinion. The frames in the same result are the evidence — when they disagree, believe the frames.",
		],
		parameters: watchSchema,
		async execute(_toolCallId, params: WatchToolInput, signal?: AbortSignal) {
			const started = Date.now();
			const apiKey = options?.apiKey ?? process.env.GEMINI_API_KEY;
			if (!apiKey) {
				throw new Error("watch needs GEMINI_API_KEY and it is not set — nothing can be watched without it.");
			}
			const { path, question, frames, model } = params;
			const useModel = model ?? defaultModel;
			const frameCount = Math.max(0, Math.min(FRAME_MAX, frames ?? FRAME_DEFAULT));

			// Fetch the bytes. A URL is pulled host-side; a path is read wherever
			// the operations point, which under virtual hands is the sandbox.
			let bytes: Buffer;
			let framePath = path;
			if (isUrl(path)) {
				const res = await fetch(path, { signal });
				if (!res.ok) throw new Error(`could not fetch ${path} (${res.status})`);
				bytes = Buffer.from(await res.arrayBuffer());
				// ffmpeg reads a URL directly, so frames still come from the source.
			} else {
				framePath = resolveReadPath(path, cwd);
				bytes = await ops.readFile(framePath);
			}
			if (bytes.length === 0) throw new Error(`${path} is empty — there is nothing to watch.`);
			if (bytes.length > MAX_BYTES) {
				throw new Error(
					`${path} is ${(bytes.length / 1024 / 1024).toFixed(0)}MB, over the ${MAX_BYTES / 1024 / 1024 / 1024}GB limit. Trim it with ffmpeg and watch the part you care about.`,
				);
			}

			const mimeType = mimeFor(path);
			// The upload and the frames are independent, so they run together.
			const [answer, cut] = await Promise.all([
				uploadAndActivate(bytes, mimeType, apiKey, signal).then((uri) =>
					askGemini(uri, mimeType, question?.trim() || DEFAULT_QUESTION, useModel, apiKey, signal),
				),
				cutFrames(ops, framePath, frameCount, tmpDir, signal).catch(() => ({
					frames: [] as { data: string; mimeType: string }[],
					note: "No stills: cutting frames failed.",
				})),
			]);

			const seconds = (Date.now() - started) / 1000;
			const header = `Watched ${path} with ${useModel}${
				cut.frames.length ? `, and cut ${cut.frames.length} stills across it` : ""
			}.`;
			const content: (TextContent | ImageContent)[] = [
				{ type: "text", text: `${header}\n\n${answer}${cut.note ? `\n\n[${cut.note}]` : ""}` },
			];
			for (const f of cut.frames) content.push({ type: "image", data: f.data, mimeType: f.mimeType });

			return {
				content,
				details: { model: useModel, seconds, frameCount: cut.frames.length, sourceBytes: bytes.length },
			};
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(formatWatchCall(args, theme));
			return text;
		},
		renderResult(result, renderOptions, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(formatWatchResult(result as any, renderOptions, theme, context.showImages));
			return text;
		},
	};
}

export function createWatchTool(cwd: string, options?: WatchToolOptions): AgentTool<typeof watchSchema> {
	return wrapToolDefinition(createWatchToolDefinition(cwd, options));
}
