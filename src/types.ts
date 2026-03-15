export type IncomingDocument = {
  id: number;
  title: string;
  content: string;
  createdAt?: string;
};

export type StoredDocument = {
  id: number;
  title: string;
  content: string;
  createdAt: string;
};

export type StoredChunk = {
  text: string;
  embedding: number[];
  documentId: number;
  documentTitle: string;
  chunkIndex: number;
};

export type ChunkItem = {
  text: string;
  embedding: number[];
  documentId: number;
  documentTitle: string;
  chunkIndex: number;
};

/** embedding 前のチャンク（/train でベクトル化前に保持） */
export type ChunkToEmbed = Omit<ChunkItem, "embedding">;

/** GET /documents 一覧API用 */
export type DocumentListItem = {
  id: number;
  title: string;
  contentLength: number;
  createdAt: string;
};

/** GET /chunks 一覧API用 */
export type ChunkListItem = {
  documentId: number;
  documentTitle: string;
  chunkIndex: number;
  textLength: number;
  text: string;
};

/** POST /delete のリクエスト body */
export type DeleteBody = {
  id?: number;
};