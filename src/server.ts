import Fastify from "fastify";
import cors from "@fastify/cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import * as db from "./db";
import type {
  ChunkToEmbed,
  DeleteBody,
  DocumentListItem,
  ChunkListItem,
  IncomingDocument,
  StoredDocument,
  StoredChunk
} from "./types";

dotenv.config();

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

const EMBEDDING_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini";
const TOP_K_CHUNKS = 5;
/** このスコア未満のチャンクは GPT に渡さず除外する（不適切な回答を減らす） */
const MIN_CHUNK_SCORE = 0.3;

const app = Fastify({
  logger: true
});

type AskBody = {
  question?: string;
};

type TrainBody = {
  documents: IncomingDocument[];
};

/**
 * 文章を指定文字数で分割する（チャンク分割）。
 * 今後の改善の中心: 見出し単位 / 段落単位 / overlap あり。日本語では精度向上に大きく寄与する。
 */
function splitText(text: string, chunkSize = 500): string[] {
  const chunks: string[] = [];

  for (let i = 0; i < text.length; i += chunkSize) {
    chunks.push(text.slice(i, i + chunkSize));
  }

  return chunks;
}

/** テキストを OpenAI Embedding でベクトル化 */
async function embedText(text: string): Promise<number[]> {
  if (!openai) throw new Error("OPENAI_API_KEY is not set");
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text
  });
  return res.data[0].embedding;
}

/** 複数テキストを一括で embedding（レート制限に配慮して順次） */
async function embedTexts(texts: string[]): Promise<number[][]> {
  if (!openai || texts.length === 0) return [];
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts
  });
  const byIndex = new Map(res.data.map((d) => [d.index, d.embedding]));
  return texts.map((_, i) => byIndex.get(i) ?? []);
}

/** コサイン類似度（ノルムで正規化） */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** クエリに似たチャンクを上位 topK 件取得（スコア付きで返す）※メモリ用 */
function searchChunksMemory(
  queryEmbedding: number[],
  topK: number
): { chunk: StoredChunk; score: number }[] {
  return [...chunks]
    .map((c) => ({ chunk: c, score: cosineSimilarity(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

async function searchChunks(
  queryEmbedding: number[],
  topK: number
): Promise<{ chunk: StoredChunk; score: number }[]> {
  if (isDbEnabled()) return db.searchChunks(queryEmbedding, topK);
  return searchChunksMemory(queryEmbedding, topK);
}

const documents: StoredDocument[] = [];
const chunks: StoredChunk[] = [];

function isDbEnabled(): boolean {
  return !!db.getPool();
}

async function listDocumentsStore(): Promise<DocumentListItem[]> {
  if (isDbEnabled()) {
    const list = await db.listDocuments();
    return list.map((d) => ({ id: d.id, title: d.title, contentLength: d.content.length, createdAt: d.createdAt }));
  }
  return documents.map((d) => ({ id: d.id, title: d.title, contentLength: d.content.length, createdAt: d.createdAt }));
}

async function getDocumentStore(id: number): Promise<StoredDocument | null> {
  if (isDbEnabled()) return db.getDocumentById(id);
  return documents.find((d) => d.id === id) ?? null;
}

async function listChunksStore(documentId?: number): Promise<ChunkListItem[]> {
  if (isDbEnabled()) return db.listChunks(documentId);
  const filtered =
    documentId === undefined ? chunks : chunks.filter((c) => c.documentId === documentId);
  return filtered.map((c) => ({
    documentId: c.documentId,
    documentTitle: c.documentTitle,
    chunkIndex: c.chunkIndex,
    textLength: c.text.length,
    text: c.text
  }));
}

app.get("/health", async () => {
  return {
    ok: true,
    service: "kudo-ai-saas",
    message: "サーバーが稼働中です"
  };
});

app.get("/documents", async () => {
  const list = await listDocumentsStore();
  return { ok: true, count: list.length, documents: list };
});

app.get<{ Params: { id: string } }>("/documents/:id", async (request, reply) => {
  const id = Number(request.params.id);
  if (!Number.isInteger(id) || id < 1) {
    return reply.status(400).send({ ok: false, error: "Invalid document id" });
  }
  const doc = await getDocumentStore(id);
  if (!doc) {
    return reply.status(404).send({ ok: false, error: "Document not found" });
  }
  return {
    ok: true,
    document: {
      id: doc.id,
      title: doc.title,
      content: doc.content,
      createdAt: doc.createdAt
    }
  };
});

app.get<{ Querystring: { documentId?: string } }>("/chunks", async (request, reply) => {
  const documentIdParam = request.query?.documentId;
  const documentId = documentIdParam != null && documentIdParam !== "" ? Number(documentIdParam) : undefined;
  if (documentId !== undefined && !Number.isInteger(documentId)) {
    return reply.status(400).send({ ok: false, error: "documentId must be an integer" });
  }

  const list = await listChunksStore(documentId);

  return {
    ok: true,
    count: list.length,
    chunks: list
  };
});

app.post("/ask", async (request, reply) => {
  const body = request.body as AskBody | undefined;
  const question = body?.question?.trim();

  if (!question) {
    return reply.status(400).send({
      ok: false,
      error: "question is required"
    });
  }

  if (!openai) {
    return reply.status(503).send({
      ok: false,
      error: "OPENAI_API_KEY is not set. Ask is unavailable."
    });
  }

  try {
    const queryEmbedding = await embedText(question);
    const topChunksWithScore = await searchChunks(queryEmbedding, TOP_K_CHUNKS);
    const relevantChunksWithScore = topChunksWithScore.filter(({ score }) => score >= MIN_CHUNK_SCORE);
    const chunksDiscarded = topChunksWithScore.length - relevantChunksWithScore.length;

    const wpBase = process.env.WP_SITE_URL ? process.env.WP_SITE_URL.replace(/\/$/, "") : "";

    if (relevantChunksWithScore.length === 0) {
      const suggestedSources =
        topChunksWithScore.length > 0
          ? topChunksWithScore.slice(0, TOP_K_CHUNKS).map(({ chunk }) => ({
              documentId: chunk.documentId,
              documentTitle: chunk.documentTitle,
              preview: chunk.text.slice(0, 120) + (chunk.text.length > 120 ? "…" : ""),
              url: wpBase ? `${wpBase}/?p=${chunk.documentId}` : ""
            }))
          : [];

      return {
        ok: true,
        question,
        answer:
          "このサイトの記事からは、はっきりした情報が見つかりませんでした。ただ、参考になりそうな記事があれば「参照元」からご確認いただけます。",
        chunksUsed: 0,
        chunksDiscarded,
        sources: suggestedSources,
        usedChunks: []
      };
    }

    const context = relevantChunksWithScore.map(({ chunk }) => chunk.text).join("\n\n---\n\n");

    const chatRes = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        {
          role: "system",
          content: `あなたは、以下のナレッジ（サイトの記事など）を参照して質問に答えるアシスタントです。
参照ナレッジに明示されている内容を優先して、わかりやすく日本語で答えてください。
完全に一致する答えが書かれていない場合でも、参照ナレッジの内容から自然に要約・案内できる範囲で答えてください。
ただし、参照ナレッジからまったく根拠が取れないことは断定せず、「このサイトの記事からは確認できませんでした」と伝えてください。
回答の最後に、必要に応じて参考になりそうな記事があることも案内してください。

【参照ナレッジ】
${context}`
        },
        { role: "user", content: question }
      ]
    });

    let answer = chatRes.choices[0]?.message?.content?.trim() ?? "";

    const usedChunks = relevantChunksWithScore.map(({ chunk, score }) => ({
      documentId: chunk.documentId,
      documentTitle: chunk.documentTitle,
      chunkIndex: chunk.chunkIndex,
      score: Math.round(score * 10000) / 10000,
      textPreview: chunk.text.slice(0, 120) + (chunk.text.length > 120 ? "…" : "")
    }));

    const sources = relevantChunksWithScore.map(({ chunk }) => ({
      documentId: chunk.documentId,
      documentTitle: chunk.documentTitle,
      preview: chunk.text.slice(0, 120) + (chunk.text.length > 120 ? "…" : ""),
      url: wpBase ? `${wpBase}/?p=${chunk.documentId}` : ""
    }));

    if (!answer) {
      if (sources.length > 0) {
        answer =
          "このサイトには次のような記事があります。気になる記事があれば、どの記事か指定していただければ詳しく説明できます。";
      } else {
        answer =
          "このサイトの記事から回答を探しましたが、うまく見つかりませんでした。別の聞き方や別の質問もお試しください。";
      }
    }

    return {
      ok: true,
      question,
      answer,
      chunksUsed: relevantChunksWithScore.length,
      chunksDiscarded,
      sources,
      usedChunks
    };
  } catch (err) {
    app.log.error(err);
    return reply.status(500).send({
      ok: false,
      error: err instanceof Error ? err.message : "Failed to generate answer"
    });
  }
});

app.post("/train", async (request, reply) => {
  app.log.info({ trainBody: request.body }, "Incoming /train body");
  const body = request.body as TrainBody | undefined;
  const incoming = body?.documents;

  if (!Array.isArray(incoming) || incoming.length === 0) {
    app.log.error({ body }, "Invalid /train payload: documents missing");
    return reply.status(400).send({
      ok: false,
      error: "documents[] (non-empty array) is required"
    });
  }

  const stored: StoredDocument[] = [];
  const textsToEmbed: ChunkToEmbed[] = [];

  for (const doc of incoming) {
    const id = doc?.id;
    const title = doc?.title?.trim();
    const content = doc?.content?.trim();

    if (id == null || !Number.isInteger(id) || id < 1) {
      return reply.status(400).send({
        ok: false,
        error: "Each document must have a valid id (positive integer)"
      });
    }
    if (!title) {
      return reply.status(400).send({
        ok: false,
        error: "Each document must have title"
      });
    }
    if (!content) {
      return reply.status(400).send({
        ok: false,
        error: "Each document must have content"
      });
    }

    const document: StoredDocument = {
      id,
      title,
      content,
      createdAt: doc.createdAt ?? new Date().toISOString()
    };

    stored.push(document);

    const sourceText = `# ${title}\n\n${content}`;
    const docChunks = splitText(sourceText);
    for (let i = 0; i < docChunks.length; i++) {
      textsToEmbed.push({ text: docChunks[i], documentId: id, documentTitle: title, chunkIndex: i });
    }
  }

  if (!openai || textsToEmbed.length === 0) {
    const reason = !openai
      ? "OPENAI_API_KEY is not set or failed to load"
      : "no chunks to embed (textsToEmbed.length === 0)";
    app.log.warn({ hasOpenAi: !!openai, chunksCount: textsToEmbed.length }, ` /train 503: ${reason}`);
    return reply.status(503).send({
      ok: false,
      error: "OPENAI_API_KEY is required for /train. Cannot store documents without embeddings."
    });
  }

  let vectors: number[][];
  try {
    const batchSize = 100;
    const allVectors: number[][] = [];
    for (let i = 0; i < textsToEmbed.length; i += batchSize) {
      const batch = textsToEmbed.slice(i, i + batchSize);
      const batchVectors = await embedTexts(batch.map((b) => b.text));
      allVectors.push(...batchVectors);
    }
    vectors = allVectors;
  } catch (err) {
    app.log.error(err);
    return reply.status(502).send({
      ok: false,
      error: err instanceof Error ? err.message : "Embedding failed"
    });
  }

  if (isDbEnabled()) {
    try {
      await db.trainInTransaction(
        stored,
        textsToEmbed.map((b, i) => ({
          text: b.text,
          embedding: vectors[i],
          documentId: b.documentId,
          documentTitle: b.documentTitle,
          chunkIndex: b.chunkIndex
        }))
      );
    } catch (err) {
      app.log.error(err);
      return reply.status(502).send({
        ok: false,
        error: err instanceof Error ? err.message : "Database transaction failed"
      });
    }
  } else {
    for (const doc of stored) {
      const existingDocIndex = documents.findIndex((d) => d.id === doc.id);
      if (existingDocIndex !== -1) documents.splice(existingDocIndex, 1);
      for (let i = chunks.length - 1; i >= 0; i--) {
        if (chunks[i].documentId === doc.id) chunks.splice(i, 1);
      }
    }
    for (const doc of stored) documents.push(doc);
    for (let i = 0; i < textsToEmbed.length; i++) {
      const b = textsToEmbed[i];
      chunks.push({
        text: b.text,
        embedding: vectors[i],
        documentId: b.documentId,
        documentTitle: b.documentTitle,
        chunkIndex: b.chunkIndex
      });
    }
  }

  return {
    ok: true,
    message: "Training data received and stored",
    count: stored.length,
    chunksAdded: textsToEmbed.length,
    documents: stored.map((d) => ({
      id: d.id,
      title: d.title,
      contentLength: d.content.length,
      createdAt: d.createdAt
    }))
  };
});

app.post("/delete", async (request, reply) => {
  const body = request.body as DeleteBody | undefined;
  const id = body?.id;

  if (id == null || !Number.isInteger(id) || id < 1) {
    return reply.status(400).send({
      ok: false,
      error: "id (positive integer) is required"
    });
  }

  if (isDbEnabled()) {
    try {
      const deleted = await db.deleteDocumentById(id);
      return {
        ok: true,
        deleted,
        id
      };
    } catch (err) {
      app.log.error(err);
      return reply.status(502).send({
        ok: false,
        error: err instanceof Error ? err.message : "Database delete failed"
      });
    }
  }

  const docIndex = documents.findIndex((d) => d.id === id);
  if (docIndex !== -1) {
    documents.splice(docIndex, 1);
  }
  for (let i = chunks.length - 1; i >= 0; i--) {
    if (chunks[i].documentId === id) {
      chunks.splice(i, 1);
    }
  }

  return {
    ok: true,
    deleted: docIndex !== -1,
    id
  };
});

const start = async () => {
  try {
    const databaseUrl = process.env.DATABASE_URL;
    if (databaseUrl) {
      await db.initDb(databaseUrl);
      app.log.info("Using PostgreSQL + pgvector");
    } else {
      app.log.info("Using in-memory store (set DATABASE_URL for persistence)");
    }

    await app.register(cors, {
      origin: true
    });

    const port = Number(process.env.PORT || 8787);

    await app.listen({
      port,
      host: "0.0.0.0"
    });

    app.log.info(`Kudo AI SaaS running on port ${port}`);
  } catch (error) {
    app.log.error(error);
    process.exit(1);
  }
};

start();