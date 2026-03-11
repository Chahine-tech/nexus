import Anthropic from "@anthropic-ai/sdk";
import { Effect } from "effect";
import { QueryExpansionError } from "../errors";

const PROMPT = (query: string) =>
	`You are a search query expander for a source code search engine. The engine indexes AST-level features: function names, type signatures, trait/interface names, crate/package/module names, stdlib symbols, and docstring tokens.

Rules:
- Output a single line of space-separated tokens — the original terms first, then 5-8 related tokens
- Prefer exact identifiers over synonyms (e.g. "tokio" not "asynchronous library", "Result" not "error type")
- Include relevant: stdlib types, popular crate names, function names, trait names, common patterns
- No explanation, no punctuation, no newlines

Examples:
Query: rust async
Expanded: rust async tokio Future Poll spawn executor runtime task Waker

Query: error handling python
Expanded: error handling python Exception try except raise ValueError TypeError traceback

Query: http client typescript
Expanded: http client typescript fetch axios request Response Headers Promise AbortController

Query: ${query}
Expanded:`;

export function expandQuery(
	query: string,
): Effect.Effect<string, QueryExpansionError> {
	const apiKey = process.env.ANTHROPIC_API_KEY;
	if (!apiKey) return Effect.succeed(query);

	// Short queries (≤2 tokens) are likely proper nouns (crate names, symbols).
	// Expansion hurts precision: generic synonyms push the exact match down.
	if (query.trim().split(/\s+/).length <= 2) return Effect.succeed(query);

	return Effect.tryPromise({
		try: async () => {
			const client = new Anthropic({ apiKey });
			const msg = await client.messages.create({
				model: "claude-haiku-4-5-20251001",
				max_tokens: 64,
				messages: [{ role: "user", content: PROMPT(query) }],
			});
			const expanded = (msg.content[0] as { text: string }).text.trim();
			return expanded.length > 0 ? expanded : query;
		},
		catch: (err) => new QueryExpansionError({ cause: err }),
	}).pipe(
		Effect.timeout("2 seconds"),
		Effect.catchAll(() => Effect.succeed(query)),
	);
}
