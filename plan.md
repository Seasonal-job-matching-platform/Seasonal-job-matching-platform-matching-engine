# Redesign: pgvector + event-driven embeddings

## Context

The current matching engine has two structural problems that surface as one symptom: **slow startup and stale recommendations**.

- The index is a single **FAISS file on disk** (`faiss.index` + 3 pickle/npy companions) at [app/recommender/embeddings_index.py](app/recommender/embeddings_index.py). On boot, the service reads it into memory ([embeddings_index.py:83-96](app/recommender/embeddings_index.py#L83-L96)). For 10k–100k jobs that load is multi-second, and it has to happen on every cold start / scale-out.
- Updates are **full rebuilds only** ([engine.py:413-433](app/recommender/engine.py#L413-L433)), triggered by a GitHub Actions cron every 6 hours ([.github/workflows/rebuild-index.yml](.github/workflows/rebuild-index.yml)). A new job posted at 12:01 is invisible until 18:00. Every rebuild re-encodes the entire catalog.
- The index files live on Azure App Service `/home/`, which is **stateful** and prevents multi-instance scale-out.

The redesign moves embeddings into Postgres via **pgvector**, makes updates **event-driven via a webhook** from the main backend, and turns the matching engine into a **stateless service**. No more index file, no more cron rebuild, no more cold-start load.

The AI's core job stays the same: encode user (interests + the existing simplified profile) → semantic-similarity match against jobs → return top-N.

## Architecture (after)

```
Main backend (job CRUD)
     │ webhook on create/update
     ▼
Matching engine  POST /jobs/{id}/embed   → encode → UPDATE jobs SET embedding = :v
                 GET  /recommend/{uid}   → encode user → SELECT … ORDER BY embedding <=> :v
                 POST /admin/backfill    → one-time backfill of jobs missing embeddings
                       │
                       ▼
              Heroku Postgres (pgvector)
              jobs.embedding vector(384), HNSW index
```

Single source of truth, no separate index file, stateless app, real-time updates.

## Database changes (Heroku Postgres)

Run as a one-off migration. **Caveat: pgvector is supported on Heroku Postgres Standard tier and up — Essential ($5/mo) does not have it.** Confirm the plan first.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedding_updated_at timestamptz;

-- HNSW for fast ANN at 10k–100k scale. cosine_ops matches our normalized vectors.
CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw
  ON jobs USING hnsw (embedding vector_cosine_ops);
```

## Code changes

### Remove

- [app/recommender/embeddings_index.py](app/recommender/embeddings_index.py) — all FAISS / pickle / npy / disk-cache logic
- [download_model.py](download_model.py) — keep (model is still needed locally)
- The startup `build_index_background()` lifespan task in [app/main.py:18-39](app/main.py#L18-L39)
- `/admin/index` and `/internal/rebuild-index` endpoints in [app/api/endpoints.py](app/api/endpoints.py)
- `/admin/reindex_jobs` in [app/api/admin.py](app/api/admin.py)
- `build_index()` in [app/recommender/engine.py:413-433](app/recommender/engine.py#L413-L433)
- [.github/workflows/rebuild-index.yml](.github/workflows/rebuild-index.yml) — no more cron
- `RECOMMENDER_CACHE_PATH`, `INDEX_PATH`, `JOB_IDS_PATH`, `JOB_TEXTS_PATH`, `VECTORS_NPY`, `REBUILD_SECRET` from [app/config.py](app/config.py)

### Add

**[app/recommender/embedder.py](app/recommender/embedder.py)** (new, small) — singleton SentenceTransformer wrapper exposing one function:
```python
def encode(text: str) -> list[float]:  # 384-dim, L2-normalized
```
Loads the model once at process start from `app/model_cache/` (already bundled). Reuses the existing model-loading logic from [embeddings_index.py:26-41](app/recommender/embeddings_index.py#L26-L41).

### Modify

**[app/recommender/engine.py](app/recommender/engine.py)**
- Keep `build_job_text_from_row()` ([engine.py:392-409](app/recommender/engine.py#L392-L409)) — reused by the embed endpoint
- Keep `build_user_profile()` ([engine.py:436-459](app/recommender/engine.py#L436-L459)) as-is (user said "just use what is already done")
- Replace `recommend_for_user()` ([engine.py:462-510](app/recommender/engine.py#L462-L510)) — drop FAISS, replace with a single SQL round-trip:
  ```sql
  SELECT id, 1 - (embedding <=> :uvec) AS score
  FROM jobs
  WHERE status = 'OPEN' AND embedding IS NOT NULL
  ORDER BY embedding <=> :uvec
  LIMIT :n;
  ```
  Then filter by `RECOMMENDER_MIN_SCORE` (0.08, unchanged). `<=>` is pgvector's cosine-distance operator; `1 - distance` recovers the cosine similarity the old code returned.
- Add `embed_job(session, job_id)` — fetches the row, builds text via `build_job_text_from_row()`, encodes, `UPDATE jobs SET embedding = :v, embedding_updated_at = now() WHERE id = :id`
- Add `backfill_embeddings(session)` — `SELECT id FROM jobs WHERE embedding IS NULL`, embeds each. Used once after migration and as a safety net.

**[app/api/endpoints.py](app/api/endpoints.py)**
- `POST /jobs/{job_id}/embed` — called by the main backend webhook on job create/update. Protected by a shared secret header. Idempotent — re-embedding is always safe.
- `POST /admin/backfill` — one-shot for jobs missing embeddings (e.g. after the initial migration, or if a webhook was missed). Background task.
- `GET /recommend/{user_id}` — same shape as today; internals now hit pgvector.

**[app/main.py](app/main.py)**
- Drop the index-loading lifespan task. Keep the embedder warmup (force-load the SentenceTransformer once on startup so the first request isn't slow) — that's ~3-5s for MiniLM from local cache, one-time, no disk index involved.

### Main-backend integration (separate repo, not this one)

On job create/update, call:
```
POST {MATCHING_ENGINE_URL}/jobs/{id}/embed
Header: X-Embed-Secret: <shared secret>
```
Fire-and-forget is fine — if it fails, `/admin/backfill` recovers it. Job delete needs no call (the row is gone).

## Critical files

- [app/recommender/engine.py](app/recommender/engine.py) — biggest changes (recommend_for_user, embed_job, backfill_embeddings)
- [app/recommender/embeddings_index.py](app/recommender/embeddings_index.py) — delete, fold model-loading into new `embedder.py`
- [app/api/endpoints.py](app/api/endpoints.py) — endpoint surface changes
- [app/main.py](app/main.py) — strip lifespan task, add embedder warmup
- [app/config.py](app/config.py) — remove cache paths, add `EMBED_SECRET`
- [app/db.py](app/db.py) — likely unchanged (already async + pooled)
- [.github/workflows/rebuild-index.yml](.github/workflows/rebuild-index.yml) — delete

## Trade-offs and risks

- **Heroku plan**: pgvector requires Standard or higher. If currently on Essential, you'll need to upgrade or this design doesn't work.
- **Webhook reliability**: if the main app misses a call, that job becomes unrecommendable. `embedding_updated_at` + the backfill endpoint is the recovery path. Could also add a nightly "embed anything older than X" pass later.
- **Vector column size**: 384 floats × 4 bytes ≈ 1.5 KB per job. 100k jobs ≈ 150 MB in the embedding column. HNSW index adds ~2-3× that. Comfortable on Standard tier.
- **Model load on cold start**: still ~3-5s for MiniLM, but that's it — no FAISS file to read, no pickle to unpack. Stateless service can scale horizontally.

## Verification

1. **Local**: spin up a Postgres with pgvector (`docker run pgvector/pgvector:pg16`), run the migration, hit `POST /admin/backfill` on a seeded jobs table, then `GET /recommend/{user_id}` and confirm results are non-empty and ordered.
2. **Compare against today's output**: pick 3-5 user IDs, call both old and new `/recommend/{uid}`, confirm the top-N overlap heavily (won't be identical — HNSW is approximate — but the same jobs should dominate the top of both lists).
3. **Webhook**: in a staging instance of the main backend, create a new job and confirm it shows up in `/recommend/{uid}` within seconds (not hours).
4. **Latency**: `/recommend` should respond in <200 ms p50 at 100k jobs with the HNSW index.
