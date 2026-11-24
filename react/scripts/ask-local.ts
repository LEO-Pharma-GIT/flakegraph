/**
 * Answer a list of questions with Ask, in process, over a graph's artifacts on
 * disk - the same agent, tools and readers the console runs, without a console,
 * a control plane or a gate. For measuring changes to the question pipeline
 * before they ship.
 *
 *   bun --conditions=react-server scripts/ask-local.ts <graph-dir> <questions.json> <out-dir>
 *       [--concurrency 8] [--only q1,q2] [--thinking-budget 4096]
 *
 * <graph-dir> is a run's artifact directory (nodes.parquet, chunks.parquet,
 * documents.parquet, run_report.json, ...). <questions.json> is a list of
 * {"id", "question"}. Each answer is written to <out-dir>/<id>.json as
 * {id, question, seconds, answer: {summary}, trace: [tool events]}; a question
 * with a saved answer is skipped, so an interrupted run resumes.
 *
 * The model comes from the usual Ask variables (FLAKEGRAPH_ASK_BASE_URL,
 * FLAKEGRAPH_ASK_API_KEY, FLAKEGRAPH_ASK_MODEL, FLAKEGRAPH_ASK_EMBEDDING_*).
 * --thinking-budget puts a local relay in front of an OpenAI-compatible base
 * URL that adds `thinking_token_budget` to every chat request, as the fleet's
 * gateway does for interactive callers, so a bare engine thinks as it would
 * behind the gateway. With the relay, FLAKEGRAPH_ASK_BASE_URL may list several
 * engines serving the same model, comma-separated; calls go round-robin.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";

interface Question {
  id: string;
  question: string;
}

function option(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index > 0 ? process.argv[index + 1] : undefined;
}

async function relayWithThinkingBudget(upstreams: string[], budget: number): Promise<string> {
  const targets = upstreams.map((upstream) => upstream.replace(/\/$/, ""));
  let turn = 0;
  const server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) {
      chunks.push(chunk as Buffer);
    }
    let body = chunks.length ? Buffer.concat(chunks).toString("utf8") : undefined;
    const route = (request.url ?? "/").replace(/^\/v1/, "");
    if (body && route.startsWith("/chat/completions")) {
      const parsed = JSON.parse(body) as Record<string, unknown>;
      parsed.thinking_token_budget ??= budget;
      body = JSON.stringify(parsed);
    }
    const headers: Record<string, string> = {};
    for (const [name, value] of Object.entries(request.headers)) {
      if (typeof value === "string" && !["host", "content-length", "connection"].includes(name)) {
        headers[name] = value;
      }
    }
    try {
      const target = targets[turn++ % targets.length]!;
      const upstreamResponse = await fetch(`${target}${route}`, { method: request.method, headers, body });
      response.writeHead(upstreamResponse.status, Object.fromEntries(upstreamResponse.headers));
      const reader = upstreamResponse.body?.getReader();
      for (let part = await reader?.read(); part && !part.done; part = await reader!.read()) {
        response.write(part.value);
      }
      response.end();
    } catch (error) {
      response.writeHead(502).end(String(error));
    }
  });
  server.requestTimeout = 0;
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
}

async function main() {
  const [graphDir, questionsPath, outDir] = process.argv.slice(2).filter((arg, index, all) => {
    return !arg.startsWith("--") && !all[index - 1]?.startsWith("--");
  });
  if (!graphDir || !questionsPath || !outDir) {
    console.error("usage: ask-local.ts <graph-dir> <questions.json> <out-dir> [--concurrency N] [--only ids] [--thinking-budget N]");
    process.exit(2);
  }
  const budget = Number.parseInt(option("thinking-budget") ?? "", 10);
  if (Number.isFinite(budget) && budget > 0 && process.env.FLAKEGRAPH_ASK_BASE_URL) {
    process.env.FLAKEGRAPH_ASK_BASE_URL = await relayWithThinkingBudget(
      process.env.FLAKEGRAPH_ASK_BASE_URL.split(","),
      budget,
    );
  }
  // Imported after the environment is final: the model resolves it once.
  const { loadLocalGraph } = await import("../src/server/graph");
  const { completeAsk } = await import("../src/server/ask/complete");

  const only = option("only")?.split(",").filter(Boolean);
  const concurrency = Math.max(1, Number.parseInt(option("concurrency") ?? "8", 10) || 8);
  const questions = (JSON.parse(await readFile(questionsPath, "utf8")) as Question[]).filter(
    (question) => !only || only.includes(question.id),
  );
  await mkdir(outDir, { recursive: true });
  const dataset = await loadLocalGraph(path.resolve(graphDir), { purpose: "retrieval" });
  console.log(`graph ${dataset.graphId}: ${dataset.documents.length} documents, ${dataset.chunks.length} chunks`);

  const pending = questions.filter((question) => {
    const file = path.join(outDir, `${question.id}.json`);
    return !existsSync(file);
  });
  let next = 0;
  async function worker() {
    while (next < pending.length) {
      const question = pending[next++]!;
      const started = Date.now();
      const trace: Record<string, unknown>[] = [];
      try {
        const answer = await completeAsk({
          dataset,
          question: question.question,
          onToolEvent: (event) => trace.push({ t: (Date.now() - started) / 1000, type: "tool", ...event }),
        });
        const seconds = (Date.now() - started) / 1000;
        await writeFile(
          path.join(outDir, `${question.id}.json`),
          JSON.stringify({ id: question.id, question: question.question, http: "200", seconds, answer, trace }, null, 1),
        );
        console.log(`${question.id} ${seconds.toFixed(0)}s ${answer.summary.length} chars`);
      } catch (error) {
        console.log(`${question.id} FAILED ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, pending.length) }, worker));
  console.log("done");
  process.exit(0);
}

void main();
