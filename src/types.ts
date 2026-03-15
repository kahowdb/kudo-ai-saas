export type IncomingDocument = {
  id: number;
  title: string;
  content: string;
  createdAt?: string;
};

export type StoredDocument = {
  siteId: string;
  id: number;
  title: string;
  content: string;
  createdAt: string;
};

export type StoredChunk = {
  siteId: string;
  text: string;
  embedding: number[];
  documentId: number;
  documentTitle: string;
  chunkIndex: number;
};

export type ChunkItem = {
  siteId: string;
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

/** リクエストで渡すサイト識別子（同一APIで複数サイトのデータを分離する） */
export const DEFAULT_SITE_ID = "default";