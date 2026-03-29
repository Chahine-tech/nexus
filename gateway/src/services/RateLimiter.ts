// Sliding-window rate limiter keyed by IP address.
//
// Each IP is allowed `maxRequests` requests per `windowMs` milliseconds.
// Old timestamps outside the current window are pruned on every check.
// The map is also pruned periodically to avoid unbounded memory growth.

export interface RateLimiterOptions {
	windowMs: number;
	maxRequests: number;
}

export class RateLimiter {
	private readonly windowMs: number;
	private readonly maxRequests: number;
	// ip → sorted list of request timestamps (ms)
	private readonly windows = new Map<string, number[]>();

	constructor({ windowMs, maxRequests }: RateLimiterOptions) {
		this.windowMs = windowMs;
		this.maxRequests = maxRequests;

		// Prune stale entries every window to keep memory bounded.
		setInterval(() => this.prune(), windowMs);
	}

	// Returns true if the request is allowed, false if rate-limited.
	check(ip: string): boolean {
		const now = Date.now();
		const cutoff = now - this.windowMs;

		const timestamps = this.windows.get(ip) ?? [];
		// Drop timestamps outside the current window.
		const active = timestamps.filter((t) => t > cutoff);
		if (active.length >= this.maxRequests) {
			this.windows.set(ip, active);
			return false;
		}
		active.push(now);
		this.windows.set(ip, active);
		return true;
	}

	private prune(): void {
		const cutoff = Date.now() - this.windowMs;
		for (const [ip, timestamps] of this.windows) {
			const active = timestamps.filter((t) => t > cutoff);
			if (active.length === 0) {
				this.windows.delete(ip);
			} else {
				this.windows.set(ip, active);
			}
		}
	}
}

// Singleton — 60 requests per minute per IP on /search.
export const rateLimiter = new RateLimiter({
	windowMs: 60_000,
	maxRequests: 60,
});
