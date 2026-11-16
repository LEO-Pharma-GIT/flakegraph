import type { GraphDataset, RunSnapshot } from "./protocol/schema";
import { graphCounts } from "./protocol/schema";
import type { QualityReport } from "./quality";

export interface InspectReport {
  graphId: string;
  graphName: string | null;
  tables: Record<string, number>;
  topEntities: Array<{ id: string; name: string; type: string; evidenceCount: number }>;
  topRelations: Array<{ id: string; source: string; target: string; relationType: string }>;
  communitySummaries: Array<{ id: string; title: string; summary: string; suggestedQuestions: string[] }>;
  zeroEntityFailure: boolean;
  gold: QualityReport["gold"];
}

export function inspectFromDataset(
  snapshot: Pick<RunSnapshot, "graphId" | "graphName">,
  dataset: GraphDataset,
  quality: QualityReport,
): InspectReport {
  const counts = graphCounts(dataset);
  const evidenceByNode = new Map<string, number>();
  for (const row of dataset.evidence) {
    const nodeId = String(row.subject_id ?? row.node_id ?? row.relation_id ?? "");
    evidenceByNode.set(nodeId, (evidenceByNode.get(nodeId) ?? 0) + 1);
  }
  const nameOf = (id: string) =>
    String(dataset.nodes.find((node) => String(node.id) === id)?.name ?? id);
  return {
    graphId: snapshot.graphId,
    graphName: snapshot.graphName,
    tables: {
      documents: counts.documents,
      nodes: counts.nodes,
      edges: counts.edges,
      communities: counts.communities,
      evidence: counts.evidence,
    },
    topEntities: [...dataset.nodes]
      .map((node) => ({
        id: String(node.id ?? ""),
        name: String(node.name ?? node.id ?? ""),
        type: String(node.primary_type ?? node.type ?? ""),
        evidenceCount: evidenceByNode.get(String(node.id ?? "")) ?? 0,
      }))
      .sort((left, right) => right.evidenceCount - left.evidenceCount)
      .slice(0, 12),
    topRelations: dataset.edges.slice(0, 12).map((edge) => ({
      id: String(edge.id ?? ""),
      source: nameOf(String(edge.source_node_id ?? edge.source ?? "")),
      target: nameOf(String(edge.target_node_id ?? edge.target ?? "")),
      relationType: String(edge.relation_type ?? ""),
    })),
    communitySummaries: dataset.communities.slice(0, 12).map((community) => ({
      id: String(community.id ?? ""),
      title: String(community.title ?? community.id ?? ""),
      summary: String(community.summary ?? ""),
      suggestedQuestions: Array.isArray(community.suggested_questions)
        ? community.suggested_questions.map(String)
        : [`What connects the ${String(community.title ?? "community")} neighborhood?`],
    })),
    zeroEntityFailure: quality.zeroEntityFailure,
    gold: quality.gold,
  };
}

export function inspectHtml(report: InspectReport): string {
  const rows = (items: string[]) => items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Inspect ${escapeHtml(report.graphName || report.graphId)}</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; color: #111; }
    h1 { font-size: 1.4rem; }
    .muted { color: #555; }
    section { margin: 1.5rem 0; }
  </style>
</head>
<body>
  <p class="muted">CLI inspect artifact · graph ${escapeHtml(report.graphId)}</p>
  <h1>${escapeHtml(report.graphName || report.graphId)}</h1>
  <p>${Object.entries(report.tables).map(([key, value]) => `${key} ${value}`).join(" · ")}</p>
  ${report.zeroEntityFailure ? "<p>0 entities after a green pipeline: treat as an OCR or schema failure.</p>" : ""}
  <section>
    <h2>Top entities</h2>
    <ul>${rows(report.topEntities.map((item) => `${item.name} (${item.type}) · ${item.evidenceCount} evidence`))}</ul>
  </section>
  <section>
    <h2>Top relations</h2>
    <ul>${rows(report.topRelations.map((item) => `${item.source} —${item.relationType}→ ${item.target}`))}</ul>
  </section>
  <section>
    <h2>Communities</h2>
    <ul>${rows(report.communitySummaries.map((item) => `${item.title}: ${item.summary}`))}</ul>
  </section>
  <section>
    <h2 id="gold-compare">Gold compare</h2>
    ${
      report.gold
        ? `<p>${escapeHtml(report.gold.goldName)} · ${report.gold.foundEntities}/${report.gold.expectedEntities} entities · ${report.gold.matchedRequired}/${report.gold.requiredTotal} required relations</p>
    <ul>${rows(
      report.gold.missingRequired.length
        ? report.gold.missingRequired.map((item) => `missing ${item.source} —${item.relationType}→ ${item.target}`)
        : ["Required gold relations are present."],
    )}</ul>`
        : "<p class=\"muted\">No gold.json matched this graph name.</p>"
    }
  </section>
</body>
</html>`;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
