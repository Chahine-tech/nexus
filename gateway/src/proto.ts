import { pack, unpack } from "msgpackr"
import type { NodeId } from "./errors"

export interface QueryRequest {
  terms: string[]
  limit: number
}

export interface NodeResult {
  id: string
  score: number
  snippet?: string
}

export interface QueryResponse {
  nodeId: NodeId
  results: NodeResult[]
}

export const encode = (msg: unknown): Uint8Array => pack(msg)
export const decode = <T>(buf: Uint8Array): T => unpack(buf) as T
