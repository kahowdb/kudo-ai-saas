import { Pool, PoolClient } from "pg";
import pgvector from "pgvector/pg";
import type { StoredDocument, StoredChunk, ChunkItem } from "./types";

let pool: Pool | null = null;

export async function initDb(databaseUrl: string): Promise<void> {
  pool = new Pool({
    connectionString: databaseUrl
  });

  const client = await pool.connect();

  try {
    await client.query("CREATE EXTENSION IF NOT EXISTS vector");
    await pgvector.registerType(client);

    await client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
      )
    `);

    await client.query(`
      CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        document_title TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding VECTOR(1536) NOT NULL,
        UNIQUE (document_id, chunk_index)
      )
    `);

    /*
    件数が増えてきたら追加検討
    CREATE INDEX chunks_embedding_cosine_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);
    */
  } finally {
    client.release();
  }
}

export function getPool(): Pool | null {
  return pool;
}

export async function listDocuments(): Promise<StoredDocument[]> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = await pool.query(`
    SELECT id, title, content, created_at
    FROM documents
    ORDER BY id DESC
  `);

  return result.rows.map((row) => ({
    id: Number(row.id),
    title: String(row.title),
    content: String(row.content),
    createdAt: new Date(row.created_at).toISOString()
  }));
}

export async function deleteDocumentById(id: number): Promise<boolean> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = await pool.query("DELETE FROM documents WHERE id = $1", [id]);
  return (result.rowCount ?? 0) > 0;
}

export async function getDocumentById(id: number): Promise<StoredDocument | null> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = await pool.query(
    `
      SELECT id, title, content, created_at
      FROM documents
      WHERE id = $1
      LIMIT 1
    `,
    [id]
  );

  if (result.rowCount === 0) {
    return null;
  }

  const row = result.rows[0];

  return {
    id: Number(row.id),
    title: String(row.title),
    content: String(row.content),
    createdAt: new Date(row.created_at).toISOString()
  };
}

export async function listChunks(
  documentId?: number
): Promise<
  { documentId: number; documentTitle: string; chunkIndex: number; textLength: number; text: string }[]
> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = documentId === undefined
    ? await pool.query(`
        SELECT document_id, document_title, chunk_index, text
        FROM chunks
        ORDER BY document_id DESC, chunk_index ASC
      `)
    : await pool.query(
        `
          SELECT document_id, document_title, chunk_index, text
          FROM chunks
          WHERE document_id = $1
          ORDER BY chunk_index ASC
        `,
        [documentId]
      );

  return result.rows.map((row) => ({
    documentId: Number(row.document_id),
    documentTitle: String(row.document_title),
    chunkIndex: Number(row.chunk_index),
    textLength: String(row.text).length,
    text: String(row.text)
  }));
}

export async function searchChunks(
  queryEmbedding: number[],
  topK: number
): Promise<{ chunk: StoredChunk; score: number }[]> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const client = await pool.connect();

  try {
    await pgvector.registerType(client);

    const result = await client.query(
      `
        SELECT
          document_id,
          document_title,
          chunk_index,
          text,
          embedding <=> $1::vector AS distance
        FROM chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
      `,
      [pgvector.toSql(queryEmbedding), topK]
    );

    return result.rows.map((row) => {
      const distance = Number(row.distance);
      const score = 1 - distance;

      return {
        chunk: {
          text: String(row.text),
          embedding: [],
          documentId: Number(row.document_id),
          documentTitle: String(row.document_title),
          chunkIndex: Number(row.chunk_index)
        },
        score
      };
    });
  } finally {
    client.release();
  }
}

export async function trainInTransaction(
  documents: StoredDocument[],
  chunkItems: ChunkItem[]
): Promise<void> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const client = await pool.connect();

  try {
    await client.query("BEGIN");
    await pgvector.registerType(client);

    for (const doc of documents) {
      await upsertDocument(client, doc);
      await deleteChunksByDocumentId(client, doc.id);
    }

    for (const chunk of chunkItems) {
      await insertChunk(client, chunk);
    }

    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

async function upsertDocument(client: PoolClient, doc: StoredDocument): Promise<void> {
  await client.query(
    `
      INSERT INTO documents (id, title, content, created_at)
      VALUES ($1, $2, $3, $4)
      ON CONFLICT (id)
      DO UPDATE SET
        title = EXCLUDED.title,
        content = EXCLUDED.content,
        created_at = EXCLUDED.created_at
    `,
    [doc.id, doc.title, doc.content, doc.createdAt]
  );
}

async function deleteChunksByDocumentId(client: PoolClient, documentId: number): Promise<void> {
  await client.query(
    `
      DELETE FROM chunks
      WHERE document_id = $1
    `,
    [documentId]
  );
}

async function insertChunk(client: PoolClient, chunk: ChunkItem): Promise<void> {
  await client.query(
    `
      INSERT INTO chunks (document_id, document_title, chunk_index, text, embedding)
      VALUES ($1, $2, $3, $4, $5::vector)
    `,
    [
      chunk.documentId,
      chunk.documentTitle,
      chunk.chunkIndex,
      chunk.text,
      pgvector.toSql(chunk.embedding)
    ]
  );
}

export type { StoredDocument, StoredChunk, ChunkItem } from "./types";