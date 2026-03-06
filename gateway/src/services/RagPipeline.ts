import Anthropic from "@anthropic-ai/sdk"
import { Effect } from "effect"
import type { RankedResult } from "./MergeEngine"

export const RAG_PROMPT = (query: string, snippets: string[]) =>
  `You are a code search assistant for a distributed AST search engine. Results are extracted AST features: function signatures, type definitions, docstrings, and code patterns from source files. Write a concise answer (3-5 sentences) that directly addresses the query. Reference specific functions, types, or patterns from the results. Be precise and technical.

Query: ${query}

Search results:
${snippets.map((s, i) => `[${i + 1}] ${s}`).join("\n")}

Answer:`

export function ragPipeline(
  query: string,
  results: RankedResult[],
): Effect.Effect<string | undefined> {
  const apiKey = process.env.ANTHROPIC_API_KEY
  if (!apiKey) return Effect.succeed(undefined)

  const snippets = results.slice(0, 5).filter((r) => r.snippet).map((r) => r.snippet!)
  if (snippets.length === 0) return Effect.succeed(undefined)

  return Effect.tryPromise({
    try: async () => {
      const client = new Anthropic({ apiKey })
      const msg = await client.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 256,
        messages: [{ role: "user", content: RAG_PROMPT(query, snippets) }],
      })
      return (msg.content[0] as { text: string }).text.trim()
    },
    catch: () => undefined,
  }).pipe(
    Effect.timeout("5 seconds"),
    Effect.catchAll(() => Effect.succeed(undefined)),
  )
}
