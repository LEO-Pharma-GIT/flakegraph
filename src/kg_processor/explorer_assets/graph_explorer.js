(() => {
  "use strict";

  const data = JSON.parse(document.getElementById("flakegraph-data").textContent);
  const graphCanvas = document.getElementById("graph-canvas");
  const palette = [
    "#087f8c", "#7a5195", "#24735b", "#bd5d38", "#32679a", "#a27013",
    "#a33f62", "#4d7c30", "#6b5ca5", "#9b4f45", "#167c6c", "#72601b",
  ];
  const tablePageSize = 50;
  const indexes = buildIndexes(data);
  const state = {
    view: "graph",
    search: "",
    nodeTypes: new Set(data.filters.node_types.map((item) => item.value)),
    relationTypes: new Set(data.filters.relation_types.map((item) => item.value)),
    communityId: "",
    minWeight: 0,
    minDegree: 0,
    evidenceOnly: false,
    hideIsolates: true,
    layout: data.layout_metadata.default,
    colorBy: "type",
    sizeBy: "degree",
    labels: "selected",
    focusNodeId: "",
    selection: null,
    visibleNodes: data.nodes.slice(),
    visibleEdges: data.edges.slice(),
    tableKind: "nodes",
    tablePage: 1,
    tableSearch: "",
    tableSortKey: "",
    tableSortDirection: 1,
  };
  let graphInitialized = false;
  let graphRenderTimer = 0;

  // Table definitions keep one rendering engine aligned with the explorer's
  // graph, evidence, community, and source records.
  const tableDefinitions = {
    nodes: {
      label: "Entities",
      rows: () => state.visibleNodes,
      columns: [
        ["name", "Name"], ["primary_type", "Type"], ["degree", "Degree"],
        ["rank", "Rank"], ["description", "Description"],
      ],
      selectionKind: "node",
    },
    edges: {
      label: "Relations",
      rows: () => state.visibleEdges.map((edge) => ({
        ...edge,
        source: nodeName(edge.source_node_id),
        target: nodeName(edge.target_node_id),
      })),
      columns: [
        ["source", "Source"], ["relation_type", "Relation"], ["target", "Target"],
        ["weight", "Weight"], ["description", "Description"],
      ],
      selectionKind: "edge",
    },
    evidence: {
      label: "Evidence",
      rows: evidenceRows,
      columns: [
        ["subject_name", "Subject"], ["subject_kind", "Kind"], ["page_number", "Page"],
        ["source", "Source"], ["quote", "Quote"],
      ],
      selectionKind: "evidence",
    },
    communities: {
      label: "Communities",
      rows: () => data.communities,
      columns: [
        ["title", "Title"], ["rating", "Rating"], ["member_count", "Members"],
        ["summary", "Summary"],
      ],
      selectionKind: "community",
      prepare: (row) => ({ ...row, member_count: arrayValue(row.member_node_ids).length }),
    },
    chunks: {
      label: "Source chunks",
      rows: () => data.chunks,
      columns: [
        ["source", "Source"], ["page_number", "Page"], ["chunk_index", "Index"],
        ["token_count", "Tokens"], ["content", "Content"],
      ],
      selectionKind: "chunk",
      prepare: (row) => ({ ...row, source: documentName(row.file_id) }),
    },
  };

  initialize();

  // Build lookup maps once. All later selection and filter work stays linear in
  // the currently visible graph rather than repeatedly scanning every table.
  function initialize() {
    renderSummary();
    renderFilterControls();
    bindControls();
    renderCommunities();
    renderDataTabs();
    renderQuality();
    renderOverview();
    updateFilteredGraph(true);
  }

  function buildIndexes(payload) {
    const nodeById = mapById(payload.nodes);
    const edgeById = mapById(payload.edges);
    const communityById = mapById(payload.communities);
    const chunkById = mapById(payload.chunks);
    const documentByFileId = new Map();
    payload.documents.forEach((document) => {
      documentByFileId.set(String(document.file_id || document.id), document);
    });
    const evidenceBySubject = groupBy(payload.evidence, "subject_id");
    const sourcesByNode = groupBy(payload.entity_sources, "node_id");
    const findingsByCommunity = groupBy(payload.community_findings, "community_id");
    const edgesByNode = new Map();
    payload.edges.forEach((edge) => {
      appendMapList(edgesByNode, String(edge.source_node_id), edge);
      appendMapList(edgesByNode, String(edge.target_node_id), edge);
    });
    return {
      nodeById, edgeById, communityById, chunkById, documentByFileId,
      evidenceBySubject, sourcesByNode, findingsByCommunity, edgesByNode,
    };
  }

  function mapById(rows) {
    return new Map(rows.map((row) => [String(row.id), row]));
  }

  function groupBy(rows, key) {
    const grouped = new Map();
    rows.forEach((row) => appendMapList(grouped, String(row[key] || ""), row));
    return grouped;
  }

  function appendMapList(map, key, value) {
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(value);
  }

  function renderSummary() {
    const qualityOk = Boolean(data.quality && data.quality.ok);
    const generated = new Date(data.generated_at);
    document.getElementById("summary-strip").innerHTML = `
      <div class="summary-identity">
        <span>Graph</span>
        <strong title="${escapeAttribute(data.graph_id)}">${escapeHtml(data.graph_id)}</strong>
        <div class="quality-status ${qualityOk ? "" : "is-failed"}">
          ${qualityOk ? "Quality checks passed" : "Quality issues found"}
        </div>
      </div>
      ${summaryStat("Entities", data.counts.nodes)}
      ${summaryStat("Relations", data.counts.edges)}
      ${summaryStat("Evidence", data.counts.evidence)}
      ${summaryStat("Communities", data.counts.communities)}
      ${summaryStat("Documents", data.counts.documents)}
      ${summaryStat("Generated", Number.isNaN(generated.valueOf()) ? "-" : generated.toLocaleDateString())}
    `;
  }

  function summaryStat(label, value) {
    return `<div class="summary-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatValue(value))}</strong></div>`;
  }

  function renderFilterControls() {
    renderCheckList("node-type-filters", data.filters.node_types, "node-type");
    renderCheckList("relation-type-filters", data.filters.relation_types, "relation-type");
    const communitySelect = document.getElementById("community-filter");
    communitySelect.innerHTML = `<option value="">All communities</option>${data.communities
      .slice()
      .sort((a, b) => String(a.title).localeCompare(String(b.title)))
      .map((community) => `<option value="${escapeAttribute(community.id)}">${escapeHtml(community.title)} (${arrayValue(community.member_node_ids).length})</option>`)
      .join("")}`;

    const weights = data.edges.map((edge) => numberValue(edge.weight));
    const degrees = data.nodes.map((node) => numberValue(node.degree));
    const weightInput = document.getElementById("weight-filter");
    const degreeInput = document.getElementById("degree-filter");
    weightInput.max = String(maximumValue(weights));
    weightInput.value = "0";
    degreeInput.max = String(maximumValue(degrees));
    degreeInput.value = "0";
    document.getElementById("layout-select").value = state.layout;
    updateRangeOutputs();
  }

  function renderCheckList(elementId, items, kind) {
    document.getElementById(elementId).innerHTML = items.map((item) => `
      <label class="check-row">
        <input type="checkbox" checked data-filter-kind="${kind}" data-value="${escapeAttribute(item.value)}">
        <span class="filter-name" title="${escapeAttribute(item.value)}">${escapeHtml(item.value)}</span>
        <span class="count-badge">${formatNumber(item.count)}</span>
      </label>
    `).join("");
  }

  function bindControls() {
    document.querySelectorAll("[data-view-target]").forEach((button) => {
      button.addEventListener("click", () => setView(button.dataset.viewTarget));
    });
    document.getElementById("graph-search").addEventListener("input", (event) => {
      state.search = event.target.value.trim().toLowerCase();
      scheduleGraphUpdate();
    });
    document.querySelectorAll("[data-filter-kind]").forEach((input) => {
      input.addEventListener("change", () => {
        const targetSet = input.dataset.filterKind === "node-type" ? state.nodeTypes : state.relationTypes;
        if (input.checked) targetSet.add(input.dataset.value);
        else targetSet.delete(input.dataset.value);
        updateFilteredGraph();
      });
    });
    document.querySelectorAll("[data-toggle-filter]").forEach((button) => {
      button.addEventListener("click", () => clearFilterGroup(button.dataset.toggleFilter));
    });
    document.getElementById("community-filter").addEventListener("change", (event) => {
      state.communityId = event.target.value;
      state.focusNodeId = "";
      updateFilteredGraph();
    });
    document.getElementById("weight-filter").addEventListener("input", (event) => {
      state.minWeight = numberValue(event.target.value);
      updateRangeOutputs();
      scheduleGraphUpdate();
    });
    document.getElementById("degree-filter").addEventListener("input", (event) => {
      state.minDegree = numberValue(event.target.value);
      updateRangeOutputs();
      scheduleGraphUpdate();
    });
    document.getElementById("evidence-filter").addEventListener("change", (event) => {
      state.evidenceOnly = event.target.checked;
      updateFilteredGraph();
    });
    document.getElementById("isolates-filter").addEventListener("change", (event) => {
      state.hideIsolates = event.target.checked;
      updateFilteredGraph();
    });
    document.getElementById("layout-select").addEventListener("change", (event) => {
      state.layout = event.target.value;
      updateFilteredGraph(true);
    });
    document.getElementById("color-select").addEventListener("change", (event) => {
      state.colorBy = event.target.value;
      renderGraph();
    });
    document.getElementById("size-select").addEventListener("change", (event) => {
      state.sizeBy = event.target.value;
      renderGraph();
    });
    document.getElementById("label-select").addEventListener("change", (event) => {
      state.labels = event.target.value;
      renderGraph();
    });
    document.getElementById("fit-graph-button").addEventListener("click", fitGraph);
    document.getElementById("reset-filters-button").addEventListener("click", resetFilters);
    document.getElementById("mobile-filter-button").addEventListener("click", () => togglePanel("filter-panel", true));
    document.getElementById("close-filter-button").addEventListener("click", () => togglePanel("filter-panel", false));
    document.getElementById("close-inspector-button").addEventListener("click", () => togglePanel("inspector-panel", false));
    document.getElementById("community-search").addEventListener("input", renderCommunities);
    document.getElementById("community-sort").addEventListener("change", renderCommunities);
    document.getElementById("community-grid").addEventListener("click", handleActionClick);
    document.getElementById("inspector-content").addEventListener("click", handleActionClick);
    document.getElementById("data-tabs").addEventListener("click", handleDataTabClick);
    document.getElementById("data-search").addEventListener("input", (event) => {
      state.tableSearch = event.target.value.trim().toLowerCase();
      state.tablePage = 1;
      renderDataTable();
    });
    document.querySelector("#data-table thead").addEventListener("click", handleTableSort);
    document.querySelector("#data-table tbody").addEventListener("click", handleTableRowClick);
    document.getElementById("previous-page-button").addEventListener("click", () => changePage(-1));
    document.getElementById("next-page-button").addEventListener("click", () => changePage(1));
    document.getElementById("download-csv-button").addEventListener("click", downloadCurrentTable);
    document.getElementById("export-view-button").addEventListener("click", exportCurrentView);
    window.addEventListener("resize", debounce(() => {
      if (state.view === "graph" && graphInitialized) window.Plotly.Plots.resize(graphCanvas);
    }, 100));
  }

  function setView(view) {
    state.view = view;
    document.body.dataset.view = view;
    document.querySelectorAll("[data-view-target]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.viewTarget === view);
    });
    document.querySelectorAll("[data-view-panel]").forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.viewPanel === view);
    });
    if (view === "graph") requestAnimationFrame(() => {
      renderGraph();
      if (graphInitialized) window.Plotly.Plots.resize(graphCanvas);
    });
    if (view === "communities") renderCommunities();
    if (view === "data") renderDataTable();
  }

  function scheduleGraphUpdate() {
    window.clearTimeout(graphRenderTimer);
    graphRenderTimer = window.setTimeout(() => updateFilteredGraph(), 90);
  }

  // Apply semantic filters before rendering so graph, table, and export views
  // always describe the same active subset.
  function updateFilteredGraph(resetViewport = false) {
    const query = state.search;
    const edgeQueryNodes = new Set();
    if (query) {
      data.edges.forEach((edge) => {
        if (edgeSearchText(edge).includes(query)) {
          edgeQueryNodes.add(String(edge.source_node_id));
          edgeQueryNodes.add(String(edge.target_node_id));
        }
      });
    }
    const communityMembers = state.communityId
      ? new Set(arrayValue(indexes.communityById.get(state.communityId)?.member_node_ids).map(String))
      : null;
    const focusMembers = state.focusNodeId ? neighborhood(state.focusNodeId) : null;
    let visibleNodes = data.nodes.filter((node) => {
      const nodeId = String(node.id);
      if (!state.nodeTypes.has(String(node.primary_type || "Unknown"))) return false;
      if (numberValue(node.degree) < state.minDegree) return false;
      if (state.evidenceOnly && !indexes.evidenceBySubject.has(nodeId)) return false;
      if (communityMembers && !communityMembers.has(nodeId)) return false;
      if (focusMembers && !focusMembers.has(nodeId)) return false;
      if (query && !nodeSearchText(node).includes(query) && !edgeQueryNodes.has(nodeId)) return false;
      return true;
    });
    let visibleNodeIds = new Set(visibleNodes.map((node) => String(node.id)));
    let visibleEdges = data.edges.filter((edge) => {
      const edgeId = String(edge.id);
      return state.relationTypes.has(String(edge.relation_type || "Unknown"))
        && numberValue(edge.weight) >= state.minWeight
        && (!state.evidenceOnly || indexes.evidenceBySubject.has(edgeId))
        && visibleNodeIds.has(String(edge.source_node_id))
        && visibleNodeIds.has(String(edge.target_node_id));
    });
    if (state.hideIsolates) {
      const connected = new Set();
      visibleEdges.forEach((edge) => {
        connected.add(String(edge.source_node_id));
        connected.add(String(edge.target_node_id));
      });
      visibleNodes = visibleNodes.filter((node) => connected.has(String(node.id)));
      visibleNodeIds = new Set(visibleNodes.map((node) => String(node.id)));
      visibleEdges = visibleEdges.filter((edge) =>
        visibleNodeIds.has(String(edge.source_node_id)) && visibleNodeIds.has(String(edge.target_node_id))
      );
    }
    state.visibleNodes = visibleNodes;
    state.visibleEdges = visibleEdges;
    renderGraph(resetViewport);
    if (state.view === "data") renderDataTable();
  }

  function nodeSearchText(node) {
    return [
      node.name,
      node.primary_type,
      node.description,
      arrayValue(node.types).join(" "),
      arrayValue(node.aliases).join(" "),
    ]
      .join(" ").toLowerCase();
  }

  function edgeSearchText(edge) {
    return [nodeName(edge.source_node_id), edge.relation_type, nodeName(edge.target_node_id), edge.description]
      .join(" ").toLowerCase();
  }

  function neighborhood(nodeId) {
    const ids = new Set([String(nodeId)]);
    (indexes.edgesByNode.get(String(nodeId)) || []).forEach((edge) => {
      ids.add(String(edge.source_node_id));
      ids.add(String(edge.target_node_id));
    });
    return ids;
  }

  // Plotly receives fixed NetworkX coordinates. Edges are grouped into a small
  // number of WebGL traces while midpoint markers retain relation selection.
  function renderGraph(resetViewport = false) {
    const empty = state.visibleNodes.length === 0;
    document.getElementById("graph-empty").hidden = !empty;
    document.getElementById("visible-count").textContent = `${formatNumber(state.visibleNodes.length)} entities · ${formatNumber(state.visibleEdges.length)} relations`;
    renderLegend();
    const positions = data.layouts[state.layout] || {};
    const traces = [];
    const edgesByType = groupRows(state.visibleEdges, (edge) => String(edge.relation_type || "Unknown"));
    Array.from(edgesByType.entries()).forEach(([relationType, edges]) => {
      const x = [];
      const y = [];
      edges.forEach((edge) => {
        const source = positions[String(edge.source_node_id)] || [0, 0];
        const target = positions[String(edge.target_node_id)] || [0, 0];
        x.push(source[0], target[0], null);
        y.push(source[1], target[1], null);
      });
      traces.push({
        type: "scattergl", mode: "lines", x, y, hoverinfo: "skip", showlegend: false,
        meta: { kind: "edge-line" },
        line: { color: colorFor(relationType, "relation"), width: edgeLineWidth(edges) },
        opacity: 0.42,
      });
    });

    if (state.visibleEdges.length) {
      traces.push({
        type: "scattergl",
        mode: "markers",
        x: state.visibleEdges.map((edge) => midpoint(positions, edge)[0]),
        y: state.visibleEdges.map((edge) => midpoint(positions, edge)[1]),
        customdata: state.visibleEdges.map((edge) => String(edge.id)),
        text: state.visibleEdges.map((edge) => `${escapeHtml(nodeName(edge.source_node_id))} — ${escapeHtml(edge.relation_type)} → ${escapeHtml(nodeName(edge.target_node_id))}<br>Weight ${escapeHtml(formatValue(edge.weight))}`),
        hovertemplate: "%{text}<extra></extra>",
        showlegend: false,
        meta: { kind: "edge" },
        marker: {
          size: state.visibleEdges.map((edge) => Math.max(7, Math.min(14, 7 + numberValue(edge.weight)))),
          color: state.visibleEdges.map((edge) => colorFor(String(edge.relation_type), "relation")),
          opacity: 0.22,
          line: { width: 0 },
        },
      });
    }

    const nodesByColor = groupRows(state.visibleNodes, nodeColorKey);
    Array.from(nodesByColor.entries()).forEach(([colorKey, nodes]) => {
      traces.push({
        type: "scattergl",
        mode: state.labels === "none" ? "markers" : "markers+text",
        x: nodes.map((node) => (positions[String(node.id)] || [0, 0])[0]),
        y: nodes.map((node) => (positions[String(node.id)] || [0, 0])[1]),
        customdata: nodes.map((node) => String(node.id)),
        text: nodes.map((node) => {
          if (state.labels === "all") return String(node.name || "");
          return state.selection?.kind === "node" && state.selection.id === String(node.id)
            ? String(node.name || "") : "";
        }),
        textfont: { color: "#31434c", size: 11 },
        textposition: "top center",
        hovertext: nodes.map((node) => `${escapeHtml(node.name)}<br>${escapeHtml(node.primary_type)} · degree ${escapeHtml(formatValue(node.degree))}<br>${escapeHtml(shortText(node.description, 180))}`),
        hovertemplate: "%{hovertext}<extra></extra>",
        showlegend: false,
        meta: { kind: "node" },
        marker: {
          size: nodes.map(nodeSize),
          color: colorFor(colorKey, state.colorBy),
          opacity: 0.93,
          line: {
            width: nodes.map((node) => state.selection?.kind === "node" && state.selection.id === String(node.id) ? 3 : 1),
            color: nodes.map((node) => state.selection?.kind === "node" && state.selection.id === String(node.id) ? "#17242b" : "#ffffff"),
          },
        },
      });
    });

    const layout = {
      autosize: true,
      margin: { l: 8, r: 8, t: 8, b: 8 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      hovermode: "closest",
      dragmode: "pan",
      showlegend: false,
      uirevision: resetViewport ? `${state.layout}-${Date.now()}` : state.layout,
      xaxis: { visible: false, zeroline: false, showgrid: false, scaleanchor: "y", scaleratio: 1 },
      yaxis: { visible: false, zeroline: false, showgrid: false },
    };
    const config = {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
      toImageButtonOptions: { format: "png", filename: `${safeFilename(data.graph_id)}-graph`, scale: 2 },
    };
    window.Plotly.react(graphCanvas, traces, layout, config).then(() => {
      if (!graphInitialized) {
        graphCanvas.on("plotly_click", handleGraphClick);
        graphInitialized = true;
      }
    });
  }

  function midpoint(positions, edge) {
    const source = positions[String(edge.source_node_id)] || [0, 0];
    const target = positions[String(edge.target_node_id)] || [0, 0];
    return [(source[0] + target[0]) / 2, (source[1] + target[1]) / 2];
  }

  function edgeLineWidth(edges) {
    const average = edges.reduce((sum, edge) => sum + numberValue(edge.weight), 0) / Math.max(1, edges.length);
    return Math.max(0.8, Math.min(3.2, 0.7 + average * 0.28));
  }

  function nodeColorKey(node) {
    if (state.colorBy === "community") return arrayValue(node.community_ids)[0] || "Unassigned";
    return String(node.primary_type || "Unknown");
  }

  function nodeSize(node) {
    if (state.sizeBy === "uniform") return 15;
    const value = state.sizeBy === "rank" ? numberValue(node.rank) : numberValue(node.degree);
    return Math.max(10, Math.min(28, 10 + Math.sqrt(Math.max(0, value)) * 4.2));
  }

  function renderLegend() {
    const groups = new Map();
    state.visibleNodes.forEach((node) => {
      const key = nodeColorKey(node);
      groups.set(key, (groups.get(key) || 0) + 1);
    });
    document.getElementById("graph-legend").innerHTML = Array.from(groups.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 14)
      .map(([key, count]) => `<span class="legend-item"><span class="legend-swatch" style="background:${colorFor(key, state.colorBy)}"></span>${escapeHtml(legendLabel(key))} <span class="count-badge">${formatNumber(count)}</span></span>`)
      .join("");
  }

  function legendLabel(key) {
    if (state.colorBy !== "community") return key;
    return indexes.communityById.get(key)?.title || key;
  }

  function handleGraphClick(event) {
    const point = event.points && event.points[0];
    if (!point || !point.data || !point.data.meta) return;
    const id = String(point.customdata || "");
    if (point.data.meta.kind === "node") selectRecord("node", id);
    if (point.data.meta.kind === "edge") selectRecord("edge", id);
  }

  function selectRecord(kind, id, openMobile = true) {
    state.selection = { kind, id: String(id) };
    if (kind === "node") renderNodeInspector(indexes.nodeById.get(String(id)));
    else if (kind === "edge") renderEdgeInspector(indexes.edgeById.get(String(id)));
    else if (kind === "community") renderCommunityInspector(indexes.communityById.get(String(id)));
    else if (kind === "evidence") renderEvidenceInspector(data.evidence.find((item) => String(item.id) === String(id)));
    else if (kind === "chunk") renderChunkInspector(indexes.chunkById.get(String(id)));
    if (openMobile && window.matchMedia("(max-width: 940px)").matches) togglePanel("inspector-panel", true);
    if (state.view === "graph") renderGraph();
  }

  // The inspector uses the same evidence/source indexes for every object type,
  // keeping provenance visible without duplicating graph data in the HTML.
  function renderOverview() {
    state.selection = null;
    document.getElementById("inspector-eyebrow").textContent = "Graph overview";
    document.getElementById("inspector-title").textContent = data.graph_id;
    const providers = data.graph_metrics.providers || {};
    document.getElementById("inspector-content").innerHTML = `
      <section class="inspector-section">
        ${detailMetrics([
          ["Entities", data.counts.nodes], ["Relations", data.counts.edges],
          ["Evidence", data.counts.evidence], ["Communities", data.counts.communities],
        ])}
      </section>
      <section class="inspector-section">
        <h3>Providers</h3>
        ${detailMetrics([
          ["OCR", providers.ocr || "-"], ["LLM", providers.llm || "-"],
          ["Embeddings", providers.embedding || "-"], ["Writer", providers.writer || "-"],
        ])}
      </section>
      <section class="inspector-section">
        <h3>Run</h3>
        <div class="description">${escapeHtml(data.run_report.run_id || "-")}</div>
      </section>
    `;
  }

  function renderNodeInspector(node) {
    if (!node) return renderOverview();
    const nodeId = String(node.id);
    const edges = (indexes.edgesByNode.get(nodeId) || []).slice().sort((a, b) => numberValue(b.weight) - numberValue(a.weight));
    const evidence = indexes.evidenceBySubject.get(nodeId) || [];
    const sources = indexes.sourcesByNode.get(nodeId) || [];
    const aliases = arrayValue(node.aliases);
    const communities = arrayValue(node.community_ids).map((id) => indexes.communityById.get(String(id))).filter(Boolean);
    document.getElementById("inspector-eyebrow").innerHTML = `<span class="type-label" style="--type-color:${colorFor(node.primary_type, "type")}">${escapeHtml(node.primary_type || "Entity")}</span>`;
    document.getElementById("inspector-title").textContent = node.name || node.id;
    document.getElementById("inspector-content").innerHTML = `
      <section class="inspector-section">
        <p class="description">${escapeHtml(node.description || "")}</p>
        ${aliases.length ? `<p><strong>Also known as</strong><br>${aliases.map((alias) => escapeHtml(alias)).join("<br>")}</p>` : ""}
      </section>
      <section class="inspector-section">
        ${detailMetrics([["Degree", node.degree], ["Rank", formatDecimal(node.rank)], ["Evidence", evidence.length], ["Sources", sources.length]])}
      </section>
      <section class="inspector-section">
        <button class="button button-secondary full-width" type="button" data-focus-node="${escapeAttribute(nodeId)}">${state.focusNodeId === nodeId ? "Clear neighborhood" : "Focus neighborhood"}</button>
      </section>
      ${recordSection("Relations", edges.map((edge) => edgeRecord(edge, nodeId)))}
      ${recordSection("Communities", communities.map(communityRecord))}
      ${evidenceSection(evidence)}
      ${sourceSection(sources)}
    `;
  }

  function renderEdgeInspector(edge) {
    if (!edge) return renderOverview();
    const evidence = indexes.evidenceBySubject.get(String(edge.id)) || [];
    const observations = arrayValue(data.edge_observations).filter((item) => String(item.edge_id) === String(edge.id));
    const sourceFiles = arrayValue(edge.source_file_ids);
    document.getElementById("inspector-eyebrow").textContent = edge.relation_type || "Relation";
    document.getElementById("inspector-title").textContent = `${nodeName(edge.source_node_id)} → ${nodeName(edge.target_node_id)}`;
    document.getElementById("inspector-content").innerHTML = `
      <section class="inspector-section"><p class="description">${escapeHtml(edge.description || "")}</p></section>
      <section class="inspector-section">${detailMetrics([["Weight", formatDecimal(edge.weight)], ["Confidence", formatDecimal(edge.confidence)], ["Assertions", observations.length || edge.evidence_count], ["Sources", sourceFiles.length || 1], ["Chunks", arrayValue(edge.source_chunk_ids).length]])}</section>
      ${sourceFiles.length ? `<section class="inspector-section"><h3>Source files</h3><p>${sourceFiles.map((fileId) => escapeHtml(documentName(fileId))).join("<br>")}</p></section>` : ""}
      <section class="inspector-section">
        <h3>Endpoints</h3>
        <div class="record-list">
          ${nodeRecord(indexes.nodeById.get(String(edge.source_node_id)))}
          ${nodeRecord(indexes.nodeById.get(String(edge.target_node_id)))}
        </div>
      </section>
      ${evidenceSection(evidence)}
    `;
  }

  function renderCommunityInspector(community) {
    if (!community) return renderOverview();
    const members = arrayValue(community.member_node_ids).map((id) => indexes.nodeById.get(String(id))).filter(Boolean);
    const findings = indexes.findingsByCommunity.get(String(community.id)) || [];
    document.getElementById("inspector-eyebrow").textContent = "Community";
    document.getElementById("inspector-title").textContent = community.title || community.id;
    document.getElementById("inspector-content").innerHTML = `
      <section class="inspector-section"><p class="description">${escapeHtml(community.summary || "")}</p></section>
      <section class="inspector-section">${detailMetrics([["Rating", formatDecimal(community.rating)], ["Members", members.length], ["Findings", findings.length], ["Level", community.level]])}</section>
      <section class="inspector-section"><h3>Rating rationale</h3><p class="description">${escapeHtml(community.rating_explanation || "")}</p></section>
      <section class="inspector-section"><button class="button button-primary full-width" type="button" data-community-graph="${escapeAttribute(community.id)}">Show in graph</button></section>
      ${recordSection("Members", members.map(nodeRecord))}
      ${recordSection("Findings", findings.map((finding) => `<div class="record-item"><strong>${escapeHtml(finding.summary || "Finding")}</strong><span>${escapeHtml(finding.explanation || "")}</span></div>`))}
      ${recordSection("Suggested questions", arrayValue(community.suggested_questions).map((question) => `<div class="record-item"><strong>${escapeHtml(question)}</strong></div>`))}
    `;
  }

  function renderEvidenceInspector(evidence) {
    if (!evidence) return renderOverview();
    const subject = evidence.subject_kind === "node" ? indexes.nodeById.get(String(evidence.subject_id)) : indexes.edgeById.get(String(evidence.subject_id));
    const chunk = indexes.chunkById.get(String(evidence.chunk_id));
    document.getElementById("inspector-eyebrow").textContent = "Evidence";
    document.getElementById("inspector-title").textContent = subject?.name || subject?.relation_type || "Grounded quote";
    document.getElementById("inspector-content").innerHTML = `
      ${evidenceSection([evidence])}
      <section class="inspector-section"><h3>Source context</h3><p class="description">${escapeHtml(chunk?.content || "")}</p></section>
    `;
  }

  function renderChunkInspector(chunk) {
    if (!chunk) return renderOverview();
    document.getElementById("inspector-eyebrow").textContent = "Source chunk";
    document.getElementById("inspector-title").textContent = documentName(chunk.file_id);
    document.getElementById("inspector-content").innerHTML = `
      <section class="inspector-section">${detailMetrics([["Page", chunk.page_number], ["Index", chunk.chunk_index], ["Tokens", chunk.token_count], ["Evidence", data.evidence.filter((item) => String(item.chunk_id) === String(chunk.id)).length]])}</section>
      <section class="inspector-section"><p class="description">${escapeHtml(chunk.content || "")}</p></section>
    `;
  }

  function detailMetrics(items) {
    return `<div class="detail-metrics">${items.map(([label, value]) => `<div class="detail-metric"><span>${escapeHtml(label)}</span><strong title="${escapeAttribute(formatValue(value))}">${escapeHtml(formatValue(value))}</strong></div>`).join("")}</div>`;
  }

  function recordSection(title, records) {
    if (!records.length) return "";
    return `<section class="inspector-section"><h3>${escapeHtml(title)}</h3><div class="record-list">${records.join("")}</div></section>`;
  }

  function evidenceSection(evidence) {
    if (!evidence.length) return "";
    return `<section class="inspector-section"><h3>Evidence</h3><div class="record-list">${evidence.map((item) => {
      const chunk = indexes.chunkById.get(String(item.chunk_id));
      return `<button class="evidence-item" type="button" data-select-kind="evidence" data-select-id="${escapeAttribute(item.id)}"><blockquote>“${escapeHtml(item.quote || "")}”</blockquote><div class="evidence-meta">${escapeHtml(documentName(item.file_id))} · page ${escapeHtml(formatValue(item.page_number || chunk?.page_number || "-"))}</div></button>`;
    }).join("")}</div></section>`;
  }

  function sourceSection(sources) {
    if (!sources.length) return "";
    return `<section class="inspector-section"><h3>Source descriptions</h3><div class="record-list">${sources.map((source) => `<div class="record-item"><strong>${escapeHtml(documentName(source.file_id))}</strong><span>${escapeHtml(source.per_file_description || "")} · ${formatNumber(source.mention_count || 0)} mentions</span></div>`).join("")}</div></section>`;
  }

  function nodeRecord(node) {
    if (!node) return "";
    return `<button class="record-item" type="button" data-select-kind="node" data-select-id="${escapeAttribute(node.id)}"><strong>${escapeHtml(node.name || node.id)}</strong><span>${escapeHtml(node.primary_type || "Entity")} · degree ${escapeHtml(formatValue(node.degree))}</span></button>`;
  }

  function edgeRecord(edge, selectedNodeId) {
    const otherId = String(edge.source_node_id) === String(selectedNodeId) ? edge.target_node_id : edge.source_node_id;
    return `<button class="record-item" type="button" data-select-kind="edge" data-select-id="${escapeAttribute(edge.id)}"><strong>${escapeHtml(edge.relation_type || "Relation")} → ${escapeHtml(nodeName(otherId))}</strong><span>Weight ${escapeHtml(formatDecimal(edge.weight))} · ${escapeHtml(shortText(edge.description, 110))}</span></button>`;
  }

  function communityRecord(community) {
    return `<button class="record-item" type="button" data-select-kind="community" data-select-id="${escapeAttribute(community.id)}"><strong>${escapeHtml(community.title || community.id)}</strong><span>Rating ${escapeHtml(formatDecimal(community.rating))} · ${formatNumber(arrayValue(community.member_node_ids).length)} members</span></button>`;
  }

  function renderCommunities() {
    const query = document.getElementById("community-search")?.value.trim().toLowerCase() || "";
    const sort = document.getElementById("community-sort")?.value || "rating";
    const communities = data.communities.filter((community) => !query || [community.title, community.summary, community.rating_explanation].join(" ").toLowerCase().includes(query));
    communities.sort((a, b) => {
      if (sort === "size") return arrayValue(b.member_node_ids).length - arrayValue(a.member_node_ids).length;
      if (sort === "title") return String(a.title).localeCompare(String(b.title));
      return numberValue(b.rating) - numberValue(a.rating);
    });
    document.getElementById("community-grid").innerHTML = communities.map((community) => {
      const findings = indexes.findingsByCommunity.get(String(community.id)) || [];
      return `<article class="community-card">
        <div class="community-card-header"><div><p class="eyebrow">Community</p><h2>${escapeHtml(community.title || community.id)}</h2></div><div class="community-rating">${escapeHtml(formatDecimal(community.rating))}</div></div>
        <p class="community-summary">${escapeHtml(community.summary || "")}</p>
        <div class="community-card-footer"><span class="community-counts">${formatNumber(arrayValue(community.member_node_ids).length)} members · ${formatNumber(findings.length)} findings</span><button class="button button-secondary" type="button" data-community-graph="${escapeAttribute(community.id)}">View graph</button></div>
      </article>`;
    }).join("");
  }

  // The data view is intentionally schema-driven: sorting, paging, filtering,
  // selection, and CSV export share the same column declarations.
  function renderDataTabs() {
    document.getElementById("data-tabs").innerHTML = Object.entries(tableDefinitions).map(([key, definition]) => `<button class="segment-button ${key === state.tableKind ? "is-active" : ""}" type="button" role="tab" data-table-kind="${key}">${escapeHtml(definition.label)}</button>`).join("");
  }

  function renderDataTable() {
    const definition = tableDefinitions[state.tableKind];
    let rows = definition.rows().map((row) => definition.prepare ? definition.prepare(row) : row);
    if (state.tableSearch) rows = rows.filter((row) => definition.columns.some(([key]) => formatValue(row[key]).toLowerCase().includes(state.tableSearch)));
    if (state.tableSortKey) rows.sort((a, b) => compareValues(a[state.tableSortKey], b[state.tableSortKey]) * state.tableSortDirection);
    const totalPages = Math.max(1, Math.ceil(rows.length / tablePageSize));
    state.tablePage = Math.min(state.tablePage, totalPages);
    const start = (state.tablePage - 1) * tablePageSize;
    const pageRows = rows.slice(start, start + tablePageSize);
    document.querySelector("#data-table thead").innerHTML = `<tr>${definition.columns.map(([key, label]) => `<th><button type="button" data-sort-key="${key}">${escapeHtml(label)}${state.tableSortKey === key ? (state.tableSortDirection === 1 ? " ↑" : " ↓") : ""}</button></th>`).join("")}</tr>`;
    document.querySelector("#data-table tbody").innerHTML = pageRows.map((row) => `<tr data-record-id="${escapeAttribute(row.id || "")}" data-record-kind="${definition.selectionKind}">${definition.columns.map(([key]) => `<td title="${escapeAttribute(formatValue(row[key]))}">${escapeHtml(formatValue(row[key]))}</td>`).join("")}</tr>`).join("");
    document.getElementById("table-count").textContent = `${formatNumber(rows.length)} records`;
    document.getElementById("page-indicator").textContent = `${state.tablePage} / ${totalPages}`;
    document.getElementById("previous-page-button").disabled = state.tablePage <= 1;
    document.getElementById("next-page-button").disabled = state.tablePage >= totalPages;
  }

  function evidenceRows() {
    const visibleIds = new Set([...state.visibleNodes, ...state.visibleEdges].map((item) => String(item.id)));
    return data.evidence.filter((item) => visibleIds.has(String(item.subject_id))).map((item) => {
      const subject = item.subject_kind === "node" ? indexes.nodeById.get(String(item.subject_id)) : indexes.edgeById.get(String(item.subject_id));
      return { ...item, subject_name: subject?.name || subject?.relation_type || item.subject_id, source: documentName(item.file_id) };
    });
  }

  function renderQuality() {
    const quality = data.quality || { ok: false, checks: [] };
    const schema = data.schema || { ok: false, checks: [] };
    const failedChecks = arrayValue(quality.checks).filter((check) => !check.ok && check.severity !== "warning").length;
    const warnings = arrayValue(quality.checks).filter((check) => !check.ok && check.severity === "warning").length;
    const failedSchema = arrayValue(schema.checks).filter((check) => !check.ok).length;
    document.getElementById("quality-overview").innerHTML = `
      <div class="quality-hero ${quality.ok && schema.ok ? "" : "is-failed"}"><strong>${quality.ok && schema.ok ? "Healthy graph" : "Review required"}</strong><span>${escapeHtml(data.graph_id)}</span></div>
      <div class="quality-stat"><span>Failed graph checks</span><strong>${formatNumber(failedChecks)}</strong></div>
      <div class="quality-stat"><span>Warnings</span><strong>${formatNumber(warnings)}</strong></div>
      <div class="quality-stat"><span>Schema issues</span><strong>${formatNumber(failedSchema)}</strong></div>
      <div class="quality-stat"><span>Trace events</span><strong>${formatNumber(data.trace_events)}</strong></div>`;
    const allChecks = [
      ...arrayValue(quality.checks).map((check) => ({ ...check, group: "Graph" })),
      ...arrayValue(schema.checks).map((check) => ({ ...check, name: `${check.table}_schema`, count: arrayValue(check.missing_columns).length, group: "Schema" })),
    ];
    document.getElementById("quality-checks").innerHTML = allChecks.map((check) => {
      const warning = !check.ok && check.severity === "warning";
      const stateClass = check.ok ? "" : warning ? "is-warning" : "is-failed";
      const stateLabel = check.ok ? "passed" : warning ? "warning" : "failed";
      return `<div class="check-result ${stateClass}"><span class="check-indicator"></span><div><strong>${escapeHtml(humanize(check.name))}</strong><span>${escapeHtml(check.group)} · ${stateLabel}</span></div><span class="count-badge">${formatNumber(check.count || 0)}</span></div>`;
    }).join("");
    const metrics = flattenMetrics(data.graph_metrics).filter((item) => typeof item.value !== "object").slice(0, 60);
    document.getElementById("metric-grid").innerHTML = metrics.map((item) => `<div class="metric-item"><span title="${escapeAttribute(item.path)}">${escapeHtml(humanize(item.path))}</span><strong>${escapeHtml(formatValue(item.value))}</strong></div>`).join("");
  }

  function flattenMetrics(value, prefix = "") {
    if (!value || typeof value !== "object" || Array.isArray(value)) return [{ path: prefix, value }];
    return Object.entries(value).flatMap(([key, item]) => flattenMetrics(item, prefix ? `${prefix}.${key}` : key));
  }

  function handleActionClick(event) {
    const selectButton = event.target.closest("[data-select-kind]");
    if (selectButton) {
      selectRecord(selectButton.dataset.selectKind, selectButton.dataset.selectId);
      return;
    }
    const communityButton = event.target.closest("[data-community-graph]");
    if (communityButton) {
      state.communityId = communityButton.dataset.communityGraph;
      document.getElementById("community-filter").value = state.communityId;
      state.focusNodeId = "";
      setView("graph");
      updateFilteredGraph(true);
      return;
    }
    const focusButton = event.target.closest("[data-focus-node]");
    if (focusButton) {
      state.focusNodeId = state.focusNodeId === focusButton.dataset.focusNode ? "" : focusButton.dataset.focusNode;
      updateFilteredGraph(true);
      renderNodeInspector(indexes.nodeById.get(focusButton.dataset.focusNode));
    }
  }

  function handleDataTabClick(event) {
    const button = event.target.closest("[data-table-kind]");
    if (!button) return;
    state.tableKind = button.dataset.tableKind;
    state.tablePage = 1;
    state.tableSortKey = "";
    document.querySelectorAll("[data-table-kind]").forEach((item) => item.classList.toggle("is-active", item === button));
    renderDataTable();
  }

  function handleTableSort(event) {
    const button = event.target.closest("[data-sort-key]");
    if (!button) return;
    if (state.tableSortKey === button.dataset.sortKey) state.tableSortDirection *= -1;
    else {
      state.tableSortKey = button.dataset.sortKey;
      state.tableSortDirection = 1;
    }
    renderDataTable();
  }

  function handleTableRowClick(event) {
    const row = event.target.closest("[data-record-id]");
    if (!row || !row.dataset.recordId) return;
    selectRecord(row.dataset.recordKind, row.dataset.recordId, false);
    if (["node", "edge"].includes(row.dataset.recordKind)) setView("graph");
  }

  function clearFilterGroup(group) {
    const nodeGroup = group === "node-types";
    const targetSet = nodeGroup ? state.nodeTypes : state.relationTypes;
    const selector = nodeGroup ? "node-type" : "relation-type";
    const inputs = document.querySelectorAll(`[data-filter-kind="${selector}"]`);
    const shouldSelectAll = targetSet.size === 0;
    targetSet.clear();
    inputs.forEach((input) => {
      input.checked = shouldSelectAll;
      if (shouldSelectAll) targetSet.add(input.dataset.value);
    });
    updateFilteredGraph();
  }

  function resetFilters() {
    state.search = "";
    state.nodeTypes = new Set(data.filters.node_types.map((item) => item.value));
    state.relationTypes = new Set(data.filters.relation_types.map((item) => item.value));
    state.communityId = "";
    state.minWeight = 0;
    state.minDegree = 0;
    state.evidenceOnly = false;
    state.hideIsolates = true;
    state.focusNodeId = "";
    document.getElementById("graph-search").value = "";
    document.getElementById("community-filter").value = "";
    document.getElementById("weight-filter").value = "0";
    document.getElementById("degree-filter").value = "0";
    document.getElementById("evidence-filter").checked = false;
    document.getElementById("isolates-filter").checked = true;
    document.querySelectorAll("[data-filter-kind]").forEach((input) => { input.checked = true; });
    updateRangeOutputs();
    renderOverview();
    updateFilteredGraph(true);
  }

  function updateRangeOutputs() {
    document.getElementById("weight-output").textContent = formatDecimal(state.minWeight);
    document.getElementById("degree-output").textContent = formatNumber(state.minDegree);
  }

  function fitGraph() {
    if (!graphInitialized) return;
    window.Plotly.relayout(graphCanvas, { "xaxis.autorange": true, "yaxis.autorange": true });
  }

  function togglePanel(id, open) {
    document.getElementById(id).classList.toggle("is-open", open);
  }

  function changePage(delta) {
    state.tablePage += delta;
    renderDataTable();
  }

  function downloadCurrentTable() {
    const definition = tableDefinitions[state.tableKind];
    const rows = definition.rows().map((row) => definition.prepare ? definition.prepare(row) : row);
    const header = definition.columns.map(([_key, label]) => csvCell(label)).join(",");
    const body = rows.map((row) => definition.columns.map(([key]) => csvCell(formatValue(row[key]))).join(",")).join("\n");
    downloadText(`${safeFilename(data.graph_id)}-${state.tableKind}.csv`, `${header}\n${body}`, "text/csv;charset=utf-8");
  }

  function exportCurrentView() {
    if (state.view === "graph") {
      window.Plotly.downloadImage(graphCanvas, { format: "png", filename: `${safeFilename(data.graph_id)}-graph`, scale: 2 });
    } else if (state.view === "data") downloadCurrentTable();
    else if (state.view === "communities") {
      downloadText(`${safeFilename(data.graph_id)}-communities.json`, JSON.stringify(data.communities, null, 2), "application/json");
    } else {
      downloadText(`${safeFilename(data.graph_id)}-quality.json`, JSON.stringify({ quality: data.quality, schema: data.schema, metrics: data.graph_metrics }, null, 2), "application/json");
    }
  }

  function downloadText(filename, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  // CSV cells are neutralized before download so graph text beginning with a
  // spreadsheet formula prefix remains inert when reviewers open the export.
  function csvCell(value) {
    let text = String(value ?? "");
    if (/^[=+\-@]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }

  function compareValues(left, right) {
    const leftNumber = Number(left);
    const rightNumber = Number(right);
    if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber;
    return formatValue(left).localeCompare(formatValue(right));
  }

  function groupRows(rows, keyFunction) {
    const groups = new Map();
    rows.forEach((row) => appendMapList(groups, keyFunction(row), row));
    return groups;
  }

  function nodeName(nodeId) {
    return indexes.nodeById.get(String(nodeId))?.name || String(nodeId || "Unknown");
  }

  function documentName(fileId) {
    const document = indexes.documentByFileId.get(String(fileId));
    const uri = String(document?.source_uri || fileId || "Unknown source");
    const clean = uri.replace(/\\/g, "/").replace(/\/$/, "");
    const name = clean.split("/").pop() || clean;
    try {
      return decodeURIComponent(name);
    } catch (_error) {
      return name;
    }
  }

  function colorFor(value, namespace) {
    return palette[Math.abs(hashString(`${namespace}:${value}`)) % palette.length];
  }

  function hashString(value) {
    let hash = 0;
    for (let index = 0; index < value.length; index += 1) hash = ((hash << 5) - hash) + value.charCodeAt(index) | 0;
    return hash;
  }

  function arrayValue(value) {
    if (Array.isArray(value)) return value;
    if (value === null || value === undefined || value === "") return [];
    return [value];
  }

  function numberValue(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function maximumValue(values) {
    return values.reduce((maximum, value) => Math.max(maximum, numberValue(value)), 0);
  }

  function formatNumber(value) {
    return new Intl.NumberFormat().format(numberValue(value));
  }

  function formatDecimal(value) {
    const number = numberValue(value);
    return Number.isInteger(number) ? String(number) : number.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  function formatValue(value) {
    if (value === null || value === undefined) return "-";
    if (Array.isArray(value)) return value.map(formatValue).join(", ");
    if (typeof value === "object") return JSON.stringify(value);
    if (typeof value === "number") return formatDecimal(value);
    return String(value);
  }

  function shortText(value, length) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    return text.length <= length ? text : `${text.slice(0, length - 1)}…`;
  }

  function humanize(value) {
    return String(value || "").split(".").pop().replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function safeFilename(value) {
    return String(value || "graph").replace(/[^a-z0-9._-]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "graph";
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[character]);
  }

  function escapeAttribute(value) {
    return escapeHtml(value);
  }

  function debounce(callback, delay) {
    let timer = 0;
    return (...args) => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => callback(...args), delay);
    };
  }
})();
