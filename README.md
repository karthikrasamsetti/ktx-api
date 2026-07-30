# ktx-api

Tiny serverless proxy that lets the `ktx` terminal on karthikrasamsetti.github.io talk to Groq
without exposing the API key in the browser.

Flow: portfolio (browser) → this function → Groq → back.

## Files
- `api/chat.js` — the proxy. Holds the system prompt (what ktx knows about Karthik) and calls Groq.
- `vercel.json` — function config.

## Deploy

1. Create a **new GitHub repo** called `ktx-api` and push these files.
2. On [vercel.com](https://vercel.com): **Add New → Project → import `ktx-api`**. Accept defaults, deploy.
3. In the project: **Settings → Environment Variables**, add:
   - Name: `GROQ_API_KEY`
   - Value: your Groq key (from console.groq.com)
   - Apply to Production, Preview, Development.
4. **Redeploy** (Deployments → ⋯ → Redeploy) so the env var takes effect.
5. Your endpoint is now: `https://ktx-api.vercel.app/api/chat` (Vercel shows the exact URL — it may have a suffix).

## Test it
```bash
curl -X POST https://YOUR-DEPLOYMENT.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"what does karthik build?"}]}'
```
Expect a JSON `{ "reply": "..." }`.

## Notes
- Only requests from `karthikrasamsetti.github.io` (and localhost) are allowed via CORS — see `ALLOWED_ORIGINS` in `api/chat.js`.
- To change what ktx knows, edit `SYSTEM_PROMPT` in `api/chat.js` and redeploy.
- Model is `llama-3.3-70b-versatile` on Groq's free tier.