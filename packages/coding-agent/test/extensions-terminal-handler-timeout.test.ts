import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { discoverAndLoadExtensions } from "../src/core/extensions/loader.ts";
import { ExtensionRunner } from "../src/core/extensions/runner.ts";
import { SessionManager } from "../src/core/session-manager.ts";

import { createInMemoryModelRegistry } from "./model-runtime-test-utils.ts";

// B12: a terminal-event handler whose I/O never resolves must not wedge the exit path.
describe("terminal event handler timeout", () => {
	let tempDir: string;

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-terminal-timeout-"));
		fs.mkdirSync(path.join(tempDir, "extensions"));
		vi.spyOn(console, "error").mockImplementation(() => {});
	});

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	async function createRunner(source: string) {
		fs.writeFileSync(path.join(tempDir, "extensions", "hang.ts"), source);
		const result = await discoverAndLoadExtensions([], tempDir, tempDir);
		const mr = await createInMemoryModelRegistry(AuthStorage.inMemory());
		return new ExtensionRunner(result.extensions, result.runtime, tempDir, SessionManager.inMemory(), mr);
	}

	const preview = { contextEntries: [], contextMessages: [], llmMessages: [], pendingMessages: [], canContinue: true };

	it("unblocks a hung turn_end handler and names it", async () => {
		const runner = await createRunner(`export default p => p.on("turn_end", () => new Promise(() => {}));`);
		const errors: string[] = [];
		runner.onError((e) => errors.push(e.error));
		vi.useFakeTimers();
		const pending = runner.emitBoundary(
			{
				type: "turn_end",
				turnIndex: 0,
				message: { role: "assistant", content: [], timestamp: 0 } as never,
				toolResults: [],
				messageEntryId: "m",
				toolResultEntryIds: [],
				outcome: "completed",
			} as never,
			() => preview,
		);
		await vi.advanceTimersByTimeAsync(60_000);
		const result = await pending;
		expect(result.continue).toBe(false);
		expect(errors.some((e) => e.includes("[pi-exit] turn_end handler") && e.includes("hang.ts"))).toBe(true);
	});

	it("unblocks a hung agent_end handler", async () => {
		const runner = await createRunner(`export default p => p.on("agent_end", () => new Promise(() => {}));`);
		vi.useFakeTimers();
		const pending = runner.emit({ type: "agent_end", messages: [] } as never);
		await vi.advanceTimersByTimeAsync(60_000);
		await expect(pending).resolves.toBeUndefined();
	});

	it("does not bound non-terminal events", async () => {
		const runner = await createRunner(
			`export default p => p.on("session_start", () => new Promise((r) => setTimeout(r, 120_000)));`,
		);
		vi.useFakeTimers();
		let settled = false;
		const pending = runner.emit({ type: "session_start", reason: "startup" } as never).then(() => {
			settled = true;
		});
		await vi.advanceTimersByTimeAsync(60_000);
		expect(settled).toBe(false);
		await vi.advanceTimersByTimeAsync(60_000);
		await pending;
		expect(settled).toBe(true);
	});
});
