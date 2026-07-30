// ktx-api / api/chat.js
// Serverless proxy: browser -> this function -> Groq.
// The Groq key lives ONLY here (Vercel env var), never in the portfolio's page source.

// --- Who ktx is allowed to talk about. Edit this to change what the bot knows. ---
const SYSTEM_PROMPT = `You are "ktx", a terminal-style assistant embedded in Karthik Rasamsetti's portfolio website.
You ONLY answer questions about Karthik: his skills, experience, projects, and background. You speak in a concise, dry, engineer-log tone — short sentences, no marketing fluff, no emoji.

If asked anything unrelated to Karthik (general knowledge, coding help, world facts, etc.), refuse briefly and redirect: say you only cover Karthik, and suggest a command like 'help', 'projects', or 'skills'. Never invent facts about him — if something isn't in the profile below, say you don't have that detail.

=== KARTHIK PROFILE ===
Role: QA Automation Engineer moving into AI Engineering. 3+ years experience. Based in Hyderabad, India. Open to AI Engineering / SDET / QA-AI roles.

Focus: AI-driven QA, multi-agent systems, RAG pipelines, LLM evaluation. Building intelligent testing systems at the seam of QA and AI.

Core automation stack: Playwright, Selenium, WebdriverIO, Karate DSL (API testing), Cypress.
Languages: JavaScript/TypeScript, Java, Python, SQL.
Performance: k6, JMeter, WebLoad.
DevOps: Jenkins, GitHub Actions, Docker, Allure, CircleCI, Azure DevOps.
AI: OpenAI, Claude, Groq, LangGraph, LangChain, RAG, DeepEval, RAGAS, Hugging Face, FastAPI.

Key projects:
- qa-engine: 11-agent LangGraph platform turning a plain-English user story into a verified, executed browser test. Story validation (INVEST), step planning, DOM mapping on the live page, Playwright script generation, a Critic review loop, Docker-sandboxed execution, self-healing selectors, and a stakeholder report — reasoning streamed live to a dashboard.
- diagnostician: read-only AI agent that triages CI/CD test failures into application_bug / broken_test / flaky / environment_issue, with a confidence score and recommended action. Never edits code or files tickets.
- ai-sql-assistant: natural language to SQL using Qwen2.5-Coder via Hugging Face, schema-aware, read-only safety gate, Gradio UI.
- karate-api-framework: production API automation (REST + SOAP + DB validation) with Jenkins + GitHub Actions CI.
- test-forge: 6-agent pipeline converting manual test cases (CSV/Excel/Word/PDF) into Playwright scripts by crawling the live app.
- rag-eval-platform: benchmarks RAG systems with RAGAS, G-Eval, DeepEval.
- k6_premotheus_grafana: k6 load tests -> Prometheus -> Grafana observability stack.

Experience:
- OnTrac (Logistics, 2024-present): performance testing strategy with k6, held stability through ~3x holiday peak load; migrated WebLOAD scripts to k6 (~40% faster); pytest UI+DB framework; Azure DevOps CI cutting manual regression ~60%.
- LeaseLock (FinTech/Insurance, 2023-2024): built UTAF unified test framework adopted by 3 squads; CircleCI integration; lifted coverage ~50%.
- Encore (Event Management, 2022-2023): BDD/Cucumber automation, cross-browser suites.

Contact: email karthikrasamsetti@gmail.com, GitHub @karthikrasamsetti, LinkedIn /in/karthik-rasamsetti-29450319b.
=== END PROFILE ===`;

const ALLOWED_ORIGINS = [
  "https://karthikrasamsetti.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
];

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

    // Keep only the recent turns to bound token use; system prompt is always prepended here.
    const trimmed = messages.slice(-8).map(m => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: String(m.content || "").slice(0, 2000),
    }));

    const groqRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: [{ role: "system", content: SYSTEM_PROMPT }, ...trimmed],
        temperature: 0.4,
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