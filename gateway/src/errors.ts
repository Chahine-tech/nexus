export type NodeId = string

export class NodeTimeoutError extends Error {
  readonly _tag = "NodeTimeoutError"
  constructor(public nodeId: NodeId, public terms: string[]) {
    super(`Node ${nodeId} timed out`)
  }
}

export class NodeDeadError extends Error {
  readonly _tag = "NodeDeadError"
  constructor(public nodeId: NodeId) {
    super(`Node ${nodeId} is dead`)
  }
}

export class DeserializationError extends Error {
  readonly _tag = "DeserializationError"
  public raw: Uint8Array

  constructor(raw: Uint8Array, cause: unknown) {
    super("Failed to deserialize message", { cause })
    this.raw = raw
  }
}
