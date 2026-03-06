import type { NodeResult } from "../proto"

export interface RankedResult {
  id: string
  score: number
  snippet?: string
}

export function reciprocalRankFusion(
  results: NodeResult[][],
  k = 60,
): RankedResult[] {
  const scores = new Map<string, number>()

  for (const nodeResults of results) {
    for (const [rank, doc] of nodeResults.entries()) {
      scores.set(doc.id, (scores.get(doc.id) ?? 0) + 1 / (k + rank + 1))
    }
  }

  return [...scores.entries()]
    .sort(([, a], [, b]) => b - a)
    .map(([id, score]) => ({ id, score }))
}
