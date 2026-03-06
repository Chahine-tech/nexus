import { Data } from "effect"

export type NodeId = string

export class NodeTimeoutError extends Data.TaggedError("NodeTimeoutError")<{
  readonly nodeId: NodeId
  readonly terms: string[]
}> {}

export class NodeDeadError extends Data.TaggedError("NodeDeadError")<{
  readonly nodeId: NodeId
}> {}

export class DeserializationError extends Data.TaggedError("DeserializationError")<{
  readonly raw: Uint8Array
  readonly cause: unknown
}> {}

export class QueryExpansionError extends Data.TaggedError("QueryExpansionError")<{
  readonly cause: unknown
}> {}
