# ktx-api (RAG)

Serverless RAG backend for the `ktx` terminal on karthikrasamsetti.github.io.

Flow at query time:
```
question -> HF embed (BGE) -> cosine search knowledge.json -> top 4 chunks -> Groq -> answer
```

## Files
- `ingest.py`    — LOCAL build script. Reads resume + web + READMEs, embeds, writes knowledge.json.
- `knowledge.json` — the committed index (output of ingest.py). Regenerate when your docs change.
- `api/chat.js`  — Vercel function: retrieval + Groq.
- `requirements.txt` — deps for ingest.py.
- `package.json` — marks the function as ESM.
- `vercel.json`  — function config.

## One-time setup

### 1. Build the index locally
```bash
pip install -r requirements.txt
mkdir docs                      # put resume.pdf and/or resume.docx in here
export HF_TOKEN=hf_xxx          # free token from huggingface.co/settings/tokens
python ingest.py
```
Edit `README_SOURCES` in ingest.py to add your GitHub README raw URLs first.
This writes `knowledge.json`.

### 2. Commit and deploy
```bash
git add knowledge.json api/chat.js package.json requirements.txt ingest.py
git commit -m "RAG: knowledge index + retrieval"
git push
```

### 3. Add env vars in Vercel
Project -> Settings -> Environment Variables:
- `GROQ_API_KEY` = your Groq key
- `HF_TOKEN`     = your Hugging Face read token
Then Redeploy.

## Test
```bash
curl -X POST https://ktx-api.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"what performance testing has Karthik done?"}]}'
```

## Updating your info
Re-run `python ingest.py`, commit the new `knowledge.json`, push. No code changes needed.