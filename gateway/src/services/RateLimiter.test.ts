import { describe, expect, test } from "bun:test";
import { RateLimiter } from "./RateLimiter";

describe("RateLimiter", () => {
	test("allows requests under the limit", () => {
		const rl = new RateLimiter({ windowMs: 60_000, maxRequests: 3 });
		expect(rl.check("1.2.3.4")).toBe(true);
		expect(rl.check("1.2.3.4")).toBe(true);
		expect(rl.check("1.2.3.4")).toBe(true);
	});

	test("blocks the request that exceeds the limit", () => {
		const rl = new RateLimiter({ windowMs: 60_000, maxRequests: 2 });
		rl.check("1.2.3.4");
		rl.check("1.2.3.4");
		expect(rl.check("1.2.3.4")).toBe(false);
	});

	test("different IPs have independent counters", () => {
		const rl = new RateLimiter({ windowMs: 60_000, maxRequests: 1 });
		rl.check("1.1.1.1");
		expect(rl.check("2.2.2.2")).toBe(true);
	});

	test("allows requests again after the window expires", async () => {
		const rl = new RateLimiter({ windowMs: 50, maxRequests: 1 });
		rl.check("1.2.3.4");
		expect(rl.check("1.2.3.4")).toBe(false);
		await new Promise((r) => setTimeout(r, 60));
		expect(rl.check("1.2.3.4")).toBe(true);
	});
});
