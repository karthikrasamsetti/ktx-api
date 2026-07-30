// ktx-api / api/chat.js  — RAG version.
// Flow: question -> HF embed -> cosine search knowledge.json -> top chunks -> Groq.
// Keys (GROQ_API_KEY, HF_TOKEN) live only in Vercel env vars.

import fs from "fs";
import path from "path";

const HF_MODEL = "BAAI/bge-small-en-v1.5";
const HF_URL = `https://router.huggingface.co/hf-inference/models/${HF_MODEL}/pipeline/feature-extraction`;
// BGE wants this prefix on the QUERY only (not on stored chunks).
const BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

const TOP_K = 4;

const ALLOWED_ORIGINS = [
  "https://karthikrasamsetti.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
];

// Load the index once per cold start (module scope = cached across warm calls).
let KB = null;
function loadKB() {
  if (KB) return KB;
  const p = path.join(process.cwd(), "knowledge.json");
  KB = JSON.parse(fs.readFileSync(p, "utf-8"));
  return KB;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

async function embedQuery(text) {
  const res = await fetch(HF_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      inputs: BGE_QUERY_PREFIX + text,   // single string, not an array
      options: { wait_for_model: true },
    }),
  });
  if (!res.ok) throw new Error(`HF ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const data = await res.json();
  // Normalize shape: HF may return a flat vector [f,f,...] or nested [[f,f,...]].
  let vec = data;
  while (Array.isArray(vec) && Array.isArray(vec[0])) vec = vec[0];
  if (!Array.isArray(vec) || typeof vec[0] !== "number") {
    throw new Error(`unexpected embedding shape: ${JSON.stringify(data).slice(0, 120)}`);
  }
  return vec;
}

const SYSTEM_PROMPT = `You are "ktx", a terminal-style assistant on Karthik Rasamsetti's portfolio.
Answer ONLY from the CONTEXT provided below, which is drawn from Karthik's resume, portfolio, and project READMEs.
Speak in a concise, dry, engineer-log tone — short sentences, no fluff, no emoji.
If the context doesn't contain the answer, say you don't have that detail and suggest asking about his skills, projects, or experience.
If asked anything unrelated to Karthik, refuse briefly and redirect.
Never invent facts that aren't in the context.`;

export default async function handler(req, res) {
  const origin = req.headers.origin || "";
  if (ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
  }
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const { messages } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "messages array required" });
    }

    const question = String(messages[messages.length - 1].content || "").slice(0, 500);
    const debug = req.query?.debug === "1" || (req.body && req.body.debug === true);

    // 1. retrieve
    let context = "";
    let retrievalError = null;
    let topScores = [];
    try {
      const kb = loadKB();
      const qvec = await embedQuery(question);
      const scored = kb.chunks
        .map(c => ({ c, score: cosine(qvec, c.embedding) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, TOP_K);
      topScores = scored.map(s => ({ source: s.c.source, score: +s.score.toFixed(3) }));
      context = scored.map((s, i) => `[${i + 1}] (${s.c.source})\n${s.c.text}`).join("\n\n");
    } catch (e) {
      retrievalError = String(e).slice(0, 200);
      context = "(retrieval unavailable)";
    }

    if (debug) {
      const kb2 = (() => { try { return loadKB(); } catch { return null; } })();
      return res.status(200).json({
        debug: true,
        question,
        kb_loaded: !!kb2,
        kb_count: kb2 ? kb2.count : null,
        kb_dim: kb2 ? kb2.dim : null,
        retrievalError,
        topScores,
        context_preview: context.slice(0, 300),
      });
    }

    // 2. build messages: system + context, then recent turns
    const history = messages.slice(-6).map(m => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: String(m.content || "").slice(0, 2000),
    }));
    const groqMessages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "system", content: `CONTEXT:\n${context}` },
      ...history,
    ];

    // 3. generate
    const groqRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: groqMessages,
        temperature: 0.3,
        max_tokens: 400,
      }),
    });
    if (!groqRes.ok) {
      const detail = await groqRes.text();
      return res.status(502).json({ error: "upstream_error", detail: detail.slice(0, 300) });
    }
    const data = await groqRes.json();
    const reply = data.choices?.[0]?.message?.content?.trim() || "(no response)";
    return res.status(200).json({ reply });
  } catch (err) {
    return res.status(500).json({ error: "server_error", detail: String(err).slice(0, 300) });
  }
}