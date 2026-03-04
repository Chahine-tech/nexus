import { Elysia } from "elysia"

export const app = new Elysia()
  .get("/health", () => ({ status: "ok" }))
  .get("/search", ({ query }) => {
    // TODO: QueryPlanner + fanOut + MergeEngine
    return { results: [] }
  })
