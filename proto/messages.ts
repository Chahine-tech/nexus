// Types shared between gateway and future TS packages
// Mirrored from proto/messages.rs (Rust)

export type NodeId = string
export type DocId = string

export interface QueryRequest {
  terms: string[]
  limit: number
}

export interface NodeResult {
  id: DocId
  score: number
  snippet?: string
}

export interface QueryResponse {
  nodeId: NodeId
  results: NodeResult[]
}

export type MessageType =
  | "QUERY_REQUEST"
  | "QUERY_RESPONSE"
  | "GOSSIP_IDF"
  | "GOSSIP_PAGERANK"
  | "DHT_FIND_NODE"
  | "DHT_STORE"
  | "INDEX_SHARD"
  | "NODE_JOIN"
  | "HEARTBEAT"

export interface NetworkMessage {
  type: MessageType
  payload: unknown
}
