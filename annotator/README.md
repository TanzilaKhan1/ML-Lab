# Annotator

An image annotation workbench with R2-backed storage, bbox / polygon / polyline / ellipse / keypoint tools, per-image review (accept/reject), and COCO + YOLO exports.

## Stack
- Next.js 16 (App Router) + React 19
- Fabric.js 7 for the canvas
- Cloudflare R2 (S3-compatible) via the AWS SDK v3 for persistence

## Folder layout
```
lib/r2.ts           # S3 client wrapper (server-only)
lib/storage.ts      # Annotation domain helpers (async, R2-backed)
app/api/*           # Route handlers for images, annotations, labels, stats, export
app/api/raw/…       # Image proxy so R2 creds never ship to the browser
scripts/seed-r2.mjs # One-shot uploader for existing images + JSON
```

Inside the R2 bucket: `raw/`, `annotations/`, `exports/`.

## Local development

1. Copy `.env.example` to `.env.local` and fill in the R2 credentials.
2. `npm install`
3. `npm run seed` — uploads `public/raw/` + `annotations/` + `exports/` into R2 (idempotent).
4. `npm run dev`, visit http://localhost:3000.

## Deploying to Render

This repo ships with a `render.yaml` Blueprint. To deploy:

1. Push to GitHub (make sure `.env.local` is NOT committed — `.gitignore` already excludes `.env*`).
2. In Render, go to **New → Blueprint** and point at the repo.
3. Render reads `render.yaml` and provisions a web service.
4. Fill in the five R2 secrets (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_ENDPOINT`) in the service's **Environment** tab. They are declared `sync: false` in the blueprint, so Render prompts for them.
5. Trigger a deploy. The build runs `npm ci && npm run build`; boot uses `npm start`.

Render injects `PORT` automatically; `next start` picks it up without extra config. The free plan works; upgrade if you need more RAM/CPU or want to avoid the cold-start penalty.

## Keyboard shortcuts
See the in-app "?" modal for the full list. Highlights: `V/B/P/L/E/K/G` (tools), `1-9` (quick label), `Arrow keys` (next/prev image), `Q/W` (accept/reject), `Ctrl+C/V/D` (copy/paste/duplicate), `Ctrl+Z` (undo), `F` (fit to view), `H` (toggle annotations).
