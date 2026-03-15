import { Pool, PoolClient } from "pg";
import pgvector from "pgvector/pg";
import type { StoredDocument, StoredChunk, ChunkItem } from "./types";

let pool: Pool | null = null;

export async function initDb(databaseUrl: string): Promise<void> {
  pool = new Pool({
    connectionString: databaseUrl,
    ssl: {
      rejectUnauthorized: false
    }
  });

  const client = await pool.connect();

  try {
    await client.query("CREATE EXTENSION IF NOT EXISTS vector");
    await pgvector.registerType(client);

    await client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        site_id TEXT NOT NULL DEFAULT 'default',
        id INTEGER NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (site_id, id)
      )
    `);

    const hasDocsSiteId = await hasColumn(client, "documents", "site_id");
    if (!hasDocsSiteId) {
      await client.query("ALTER TABLE documents ADD COLUMN site_id TEXT NOT NULL DEFAULT 'default'");
      await client.query("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey");
      await client.query("ALTER TABLE documents ADD PRIMARY KEY (site_id, id)");
    }

    await client.query(`
      CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        site_id TEXT NOT NULL DEFAULT 'default',
        document_id INTEGER NOT NULL,
        document_title TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding VECTOR(1536) NOT NULL,
        UNIQUE (site_id, document_id, chunk_index),
        FOREIGN KEY (site_id, document_id) REFERENCES documents(site_id, id) ON DELETE CASCADE
      )
    `);

    const hasChunksSiteId = await hasColumn(client, "chunks", "site_id");
    if (!hasChunksSiteId) {
      await client.query("ALTER TABLE chunks ADD COLUMN site_id TEXT NOT NULL DEFAULT 'default'");
      const fkName = await getChunksDocumentIdFkName(client);
      if (fkName) await client.query(`ALTER TABLE chunks DROP CONSTRAINT IF EXISTS "${fkName}"`);
      const uqName = await getChunksUniqueConstraintName(client);
      if (uqName) await client.query(`ALTER TABLE chunks DROP CONSTRAINT IF EXISTS "${uqName}"`);
      await client.query(`
        ALTER TABLE chunks ADD CONSTRAINT chunks_site_document_fkey
        FOREIGN KEY (site_id, document_id) REFERENCES documents(site_id, id) ON DELETE CASCADE
      `);
      await client.query(`
        ALTER TABLE chunks ADD CONSTRAINT chunks_site_doc_index_key UNIQUE (site_id, document_id, chunk_index)
      `);
    }
  } finally {
    client.release();
  }
}

async function hasColumn(client: PoolClient, table: string, column: string): Promise<boolean> {
  const r = await client.query(
    `SELECT 1 FROM information_schema.columns WHERE table_name = $1 AND column_name = $2`,
    [table, column]
  );
  return r.rowCount !== null && r.rowCount > 0;
}

async function getChunksDocumentIdFkName(client: PoolClient): Promise<string | null> {
  const r = await client.query(
    `SELECT constraint_name FROM information_schema.table_constraints
     WHERE table_name = 'chunks' AND constraint_type = 'FOREIGN KEY'`
  );
  return r.rows[0] ? String(r.rows[0].constraint_name) : null;
}

async function getChunksUniqueConstraintName(client: PoolClient): Promise<string | null> {
  const r = await client.query(
    `SELECT constraint_name FROM information_schema.table_constraints
     WHERE table_name = 'chunks' AND constraint_type = 'UNIQUE'`
  );
  return r.rows[0] ? String(r.rows[0].constraint_name) : null;
}

export function getPool(): Pool | null {
  return pool;
}

export async function listDocuments(siteId: string): Promise<StoredDocument[]> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = await pool.query(
    `
    SELECT site_id, id, title, content, created_at
    FROM documents
    WHERE site_id = $1
    ORDER BY id DESC
  `,
    [siteId]
  );

  return result.rows.map((row) => ({
    siteId: String(row.site_id),
    id: Number(row.id),
    title: String(row.title),
    content: String(row.content),
    createdAt: new Date(row.created_at).toISOString()
  }));
}

export async function deleteDocumentById(siteId: string, id: number): Promise<boolean> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = await pool.query("DELETE FROM documents WHERE site_id = $1 AND id = $2", [
    siteId,
    id
  ]);
  return (result.rowCount ?? 0) > 0;
}

export async function getDocumentById(
  siteId: string,
  id: number
): Promise<StoredDocument | null> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result = await pool.query(
    `
      SELECT site_id, id, title, content, created_at
      FROM documents
      WHERE site_id = $1 AND id = $2
      LIMIT 1
    `,
    [siteId, id]
  );

  if (result.rowCount === 0) {
    return null;
  }

  const row = result.rows[0];

  return {
    siteId: String(row.site_id),
    id: Number(row.id),
    title: String(row.title),
    content: String(row.content),
    createdAt: new Date(row.created_at).toISOString()
  };
}

export async function listChunks(
  siteId: string,
  documentId?: number
): Promise<
  { documentId: number; documentTitle: string; chunkIndex: number; textLength: number; text: string }[]
> {
  if (!pool) {
    throw new Error("Database is not initialized");
  }

  const result =
    documentId === undefined
      ? await pool.query(
          `
        SELECT document_id, document_title, chunk_index, text
        FROM chunks
        WHERE site_id = $1
        ORDER BY document_id DESC, chunk_index ASC
      `,
          [siteId]
        )
      : await pool.query(
          `
          SELECT document_id, document_title, chunk_index, text
          FROM chunks
          WHERE site_id = $1 AND document_id = $2
          ORDER BY chunk_index ASC
        `,
          [siteId, documentId]
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
  siteId: string,
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
          site_id,
          document_id,
          document_title,
          chunk_index,
          text,
          embedding <=> $1::vector AS distance
        FROM chunks
        WHERE site_id = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
      `,
      [pgvector.toSql(queryEmbedding), siteId, topK]
    );

    return result.rows.map((row) => {
      const distance = Number(row.distance);
      const score = 1 - distance;

      return {
        chunk: {
          siteId: String(row.site_id),
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
      await deleteChunksByDocumentId(client, doc.siteId, doc.id);
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
      INSERT INTO documents (site_id, id, title, content, created_at)
      VALUES ($1, $2, $3, $4, $5)
      ON CONFLICT (site_id, id)
      DO UPDATE SET
        title = EXCLUDED.title,
        content = EXCLUDED.content,
        created_at = EXCLUDED.created_at
    `,
    [doc.siteId, doc.id, doc.title, doc.content, doc.createdAt]
  );
}

async function deleteChunksByDocumentId(
  client: PoolClient,
  siteId: string,
  documentId: number
): Promise<void> {
  await client.query(
    `
      DELETE FROM chunks
      WHERE site_id = $1 AND document_id = $2
    `,
    [siteId, documentId]
  );
}

async function insertChunk(client: PoolClient, chunk: ChunkItem): Promise<void> {
  await client.query(
    `
      INSERT INTO chunks (site_id, document_id, document_title, chunk_index, text, embedding)
      VALUES ($1, $2, $3, $4, $5, $6::vector)
    `,
    [
      chunk.siteId,
      chunk.documentId,
      chunk.documentTitle,
      chunk.chunkIndex,
      chunk.text,
      pgvector.toSql(chunk.embedding)
    ]
  );
}

export type { StoredDocument, StoredChunk, ChunkItem } from "./types";