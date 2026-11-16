"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronDown, HelpCircle, SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { CANVAS_NODE_LIMIT, capGraph, filterGraph, type GraphFilters } from "@/server/graph-filter";
import type { GraphDataset } from "@/server/protocol/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GraphCanvas } from "@/components/console/graph-canvas";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { FacetPicker } from "@/components/console/facet-picker";
import type { ExploreFocus } from "@/components/console/graph-data-tables";
import { communityMembership } from "@/server/graph-filter";
import {
  ENTITY_NOUN,
  ENTITY_TYPE_NOUN,
  NEIGHBORHOOD_NOUN,
  RELATION_NOUN,
  RELATION_TYPE_NOUN,
  filterSummary,
  graphFacetOptions,
  prettyLabel,
  rankFacetOptions,
  type FacetOption,
} from "@/lib/graph-facets";
import {
  colorForCommunity,
  colorForType,
  findShortestPath,
  graphEdgeEnds,
  graphEdgeId,
  graphNodeId,
  graphNodeLabel,
  hopNeighborhood,
  type ColorMode,
  type FocusDepth,
  type GraphHover,
  type LayoutMode,
} from "@/lib/graph-geometry";
import { cn } from "@/lib/utils";
import {
  documentNameIndex,
  evidenceDocumentId,
  evidenceEntityId,
  evidenceRelationId,
  isClaimed,
  quantityLine,
  statedTypes,
} from "@/lib/evidence";

/** The overview draws only the best-connected core; a focus widens the cap. */
const OVERVIEW_NODE_LIMIT = 400;

/**
 * A canvas legend names this many colours before folding the rest behind
 * "+N more": eight swatches read at a glance, and the palette repeats past
 * that anyway.
 */
const LEGEND_LIMIT = 8;

export function GraphExplorer({
  dataset,
  perspectives = [],
  analyst = false,
  missingGold = [],
  goldPending = false,
  goldPresent = false,
  initialSearch = "",
  onPin,
  onFocusChange,
  jump = null,
  neighborhoodJump = null,
  clearFiltersNonce = 0,
}: {
  dataset: GraphDataset;
  perspectives?: Array<{
    id: string;
    name: string;
    search: string;
    communityIds: string[];
    lifecycle: string;
    suggestedQuestions?: string[];
  }>;
  analyst?: boolean;
  missingGold?: Array<{ id: string; source: string; target: string; relationType: string }>;
  goldPending?: boolean;
  goldPresent?: boolean;
  initialSearch?: string;
  onPin?: (targetId: string, kind: "node" | "edge") => void;
  /**
   * What the filters leave, told to whoever draws the graph elsewhere: the
   * Data tab's tables and the Export menu read the same focus this canvas
   * draws, so a filter set here narrows them too.
   */
  onFocusChange?: (focus: ExploreFocus) => void;
  /** A record chosen outside the canvas (a Data row): select it and fly there. */
  jump?: { id: string; nonce: number } | null;
  /** A neighborhood chosen outside the canvas: make it the only one shown. */
  neighborhoodJump?: { communityId: string; nonce: number } | null;
  /** Bumped when the Data tab asks for the filters to go. */
  clearFiltersNonce?: number;
}) {
  const [search, setSearch] = useState(initialSearch);
  // A jump from the palette replaces whatever was typed.
  const [seededWith, setSeededWith] = useState(initialSearch);
  if (seededWith !== initialSearch) {
    setSeededWith(initialSearch);
    setSearch(initialSearch);
  }
  const [nodeTypes, setNodeTypes] = useState<string[]>([]);
  const [relationTypes, setRelationTypes] = useState<string[]>([]);
  const [communityIds, setCommunityIds] = useState<string[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [includeIsolates, setIncludeIsolates] = useState(true);
  const [includeClaimed, setIncludeClaimed] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showNeighborhood, setShowNeighborhood] = useState(false);
  const [perspectiveId, setPerspectiveId] = useState<string>("");
  // Empty until named: a pre-filled name invites typing into the middle of
  // it, and a perspective is worth naming for what it shows.
  const [perspectiveName, setPerspectiveName] = useState("");
  const [colorMode, setColorMode] = useState<ColorMode>("type");
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("force");
  const [focusDepth, setFocusDepth] = useState<FocusDepth>(null);
  const [showMinimap, setShowMinimap] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [hiddenTypes, setHiddenTypes] = useState<string[]>([]);
  const [pathEndpoints, setPathEndpoints] = useState<string[]>([]);
  const [hover, setHover] = useState<GraphHover | null>(null);
  const [flashNonce, setFlashNonce] = useState(0);
  // The chips above the canvas read the workspace; a saved or promoted
  // perspective has to reach them without a reload.
  const utils = trpc.useUtils();
  const savePerspective = trpc.workspace.perspective.useMutation({
    onSuccess: async () => {
      toast.success("Perspective saved as draft. Publish to production after gold is green.");
      await utils.workspace.get.invalidate();
    },
    onError: (error) => toast.error(error.message),
  });
  const promotePerspective = trpc.workspace.promotePerspective.useMutation({
    onSuccess: async () => {
      toast.success("Perspective lifecycle updated.");
      await utils.workspace.get.invalidate();
    },
    onError: (error) => toast.error(error.message),
  });
  const namedPerspectives = analyst
    ? perspectives.filter((item) => item.lifecycle === "production")
    : perspectives;
  const facets = useMemo(() => graphFacetOptions(dataset), [dataset]);
  const neighborhoods = facets.neighborhoods;
  const membership = useMemo(() => communityMembership(dataset.communities ?? []), [dataset.communities]);
  const documentNames = useMemo(() => documentNameIndex(dataset.documents ?? []), [dataset.documents]);
  const filters: GraphFilters = useMemo(
    () => ({
      search,
      nodeTypes,
      relationTypes,
      communityIds,
      minimumConfidence: confidence,
      includeIsolates,
      includeClaimed,
      // The whole match stays available for neighborhoods and exports; only
      // the canvas is capped below.
      limit: Number.POSITIVE_INFINITY,
    }),
    [search, nodeTypes, relationTypes, communityIds, confidence, includeIsolates, includeClaimed],
  );
  const claimedCount = useMemo(() => dataset.edges.filter(isClaimed).length, [dataset.edges]);
  const filtered = useMemo(() => filterGraph(dataset, filters), [dataset, filters]);
  const selectedNode = dataset.nodes.find((node) => String(node.id) === selectedId);
  const selectedEdge = dataset.edges.find((edge) => String(edge.id) === selectedId);
  const focused =
    showNeighborhood ||
    search.trim().length >= 2 ||
    nodeTypes.length > 0 ||
    relationTypes.length > 0 ||
    communityIds.length > 0;
  const focusedGraph = useMemo(() => {
    if (!showNeighborhood || !selectedId) {
      return filtered;
    }
    return hopNeighborhood(filtered.nodes, filtered.edges, selectedId);
  }, [filtered, selectedId, showNeighborhood]);
  const canvasLimit = focused ? CANVAS_NODE_LIMIT : OVERVIEW_NODE_LIMIT;
  const canvas = useMemo(
    () => capGraph(focusedGraph.nodes, focusedGraph.edges, canvasLimit),
    [canvasLimit, focusedGraph.edges, focusedGraph.nodes],
  );
  const pathHighlight = useMemo(() => {
    if (pathEndpoints.length !== 2) {
      return null;
    }
    const [src, dst] = pathEndpoints;
    return findShortestPath(
      canvas.edges.map((edge, index) => ({ id: graphEdgeId(edge, index), ...graphEdgeEnds(edge) })),
      src ?? "",
      dst ?? "",
    );
  }, [canvas.edges, pathEndpoints]);
  // Ranked like the Filters facet, so the legend and the picker agree on
  // which types matter most.
  const typeLegend = useMemo(() => {
    const counts = new Map<string, number>();
    for (const node of canvas.nodes) {
      const type = String(node.primary_type ?? node.type ?? "Unknown");
      counts.set(type, (counts.get(type) ?? 0) + 1);
    }
    return rankFacetOptions([...counts.entries()].map(([id, count]) => ({ id, label: prettyLabel(id), count })));
  }, [canvas.nodes]);
  const hiddenTypeSet = useMemo(() => new Set(hiddenTypes), [hiddenTypes]);
  const hoverNode = hover?.kind === "node" ? dataset.nodes.find((node) => graphNodeId(node) === hover.id) : null;
  const hoverEdge = hover?.kind === "edge" ? dataset.edges.find((edge, index) => graphEdgeId(edge, index) === hover.id) : null;
  const pathNames = pathEndpoints.map((id) => {
    const node = dataset.nodes.find((item) => graphNodeId(item) === id);
    return node ? graphNodeLabel(node) : id;
  });
  const cardSummary = filterSummary({
    nodeTypes,
    relationTypes,
    communityIds,
    minimumConfidence: confidence,
    includeIsolates,
    includeClaimed,
  });
  const filtersActive =
    search.trim().length > 0 ||
    nodeTypes.length > 0 ||
    relationTypes.length > 0 ||
    communityIds.length > 0 ||
    confidence > 0 ||
    showNeighborhood ||
    hiddenTypes.length > 0 ||
    pathEndpoints.length > 0 ||
    focusDepth != null ||
    !includeIsolates ||
    !includeClaimed ||
    Boolean(perspectiveId);

  useEffect(() => {
    onFocusChange?.({ nodes: focusedGraph.nodes, edges: focusedGraph.edges, summary: cardSummary });
  }, [cardSummary, focusedGraph.edges, focusedGraph.nodes, onFocusChange]);
  // Requests from outside the canvas arrive as counters and are answered
  // once, during render, the way the palette's seed is applied above.
  const [jumpTaken, setJumpTaken] = useState(0);
  if (jump && jump.nonce !== jumpTaken) {
    setJumpTaken(jump.nonce);
    setSelectedId(jump.id);
    setFlashNonce((current) => current + 1);
  }
  const [neighborhoodJumpTaken, setNeighborhoodJumpTaken] = useState(0);
  if (neighborhoodJump && neighborhoodJump.nonce !== neighborhoodJumpTaken) {
    setNeighborhoodJumpTaken(neighborhoodJump.nonce);
    setCommunityIds([neighborhoodJump.communityId]);
    setSelectedId(null);
    setShowNeighborhood(false);
  }
  const [clearedAt, setClearedAt] = useState(0);
  if (clearFiltersNonce !== clearedAt) {
    setClearedAt(clearFiltersNonce);
    clearFilters();
  }

  function clearFilters() {
    setSearch("");
    setNodeTypes([]);
    setRelationTypes([]);
    setCommunityIds([]);
    setConfidence(0);
    setIncludeIsolates(true);
    setIncludeClaimed(true);
    setShowNeighborhood(false);
    setPerspectiveId("");
    setSelectedId(null);
    setFocusDepth(null);
    setHiddenTypes([]);
    setPathEndpoints([]);
  }

  function toggleHiddenType(type: string) {
    setHiddenTypes((current) => (current.includes(type) ? current.filter((item) => item !== type) : [...current, type]));
  }

  function onShiftClickNode(id: string) {
    setPathEndpoints((current) => {
      if (current.includes(id)) {
        return current.filter((item) => item !== id);
      }
      if (current.length >= 2) {
        return [id];
      }
      return [...current, id];
    });
  }

  // How many conditions narrow the graph, for the badge on the Filters button.
  const activeFilterCount =
    (nodeTypes.length > 0 ? 1 : 0) +
    (relationTypes.length > 0 ? 1 : 0) +
    (communityIds.length > 0 ? 1 : 0) +
    (confidence > 0 ? 1 : 0) +
    (includeIsolates ? 0 : 1) +
    (includeClaimed ? 0 : 1);
  const scopeStat =
    focusedGraph.nodes.length > 0
      ? `${canvas.nodes.length.toLocaleString()} of ${canvas.totalNodes.toLocaleString()} entities · ${canvas.edges.length.toLocaleString()} ${
          canvas.edges.length === 1 ? "relation" : "relations"
        }${showNeighborhood && selectedId ? " in the selected neighborhood" : ""}`
      : "No entities in this filter";
  const scopeExplanation =
    canvas.totalNodes > canvas.nodes.length
      ? "The best-connected entities are drawn; search, a filter or a neighborhood focuses the rest. Summary metrics cover the full graph."
      : "Everything the filters leave is drawn. Summary metrics cover the full graph.";

  return (
    <div className="space-y-3">
      {/*
        One toolbar for the canvas: what to look for, what narrows it, how it
        is drawn, and what that leaves - in a row rather than a stack, so the
        graph starts where the page does.
      */}
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-card px-2 py-1.5" data-testid="explore-toolbar">
        <Input
          aria-label="Search entities"
          className="h-8 w-full min-w-[12rem] sm:w-64"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Entity name or description"
        />
        <Popover open={filtersOpen} onOpenChange={setFiltersOpen}>
          <PopoverTrigger asChild>
            <Button
              size="sm"
              variant={activeFilterCount > 0 ? "secondary" : "outline"}
              className="h-8 gap-1.5"
              aria-label="Filters"
              title={cardSummary ?? "Narrow the graph by type, relation, neighborhood or confidence"}
            >
              <SlidersHorizontal className="size-3.5" aria-hidden="true" />
              Filters
              {activeFilterCount > 0 ? (
                <span className="rounded-full bg-primary px-1.5 text-[11px] font-semibold text-primary-foreground" data-testid="filters-count">
                  {activeFilterCount}
                </span>
              ) : null}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[36rem] max-w-[calc(100vw-2rem)] p-0" data-testid="explore-filters">
            <div className="flex items-center gap-3 border-b border-border px-3 py-2 text-sm" data-testid="filters-summary">
              <span className="font-medium">Filters</span>
              {cardSummary ? <span className="min-w-0 truncate text-muted-foreground">{cardSummary}</span> : null}
              {cardSummary ? (
                <Button size="sm" variant="ghost" className="ml-auto h-6 px-2 text-xs" onClick={clearFilters}>
                  Clear filters
                </Button>
              ) : null}
            </div>
            <div className="grid gap-4 p-3">
              <FacetPicker
                label="Entity types"
                noun={ENTITY_TYPE_NOUN}
                unit={ENTITY_NOUN}
                options={facets.nodeTypes}
                value={nodeTypes}
                onChange={setNodeTypes}
                testId="facet-entity-types"
              />
              <FacetPicker
                label="Relation types"
                noun={RELATION_TYPE_NOUN}
                unit={RELATION_NOUN}
                options={facets.relationTypes}
                value={relationTypes}
                onChange={setRelationTypes}
                testId="facet-relation-types"
              />
              <FacetPicker
                label="Neighborhoods"
                noun={NEIGHBORHOOD_NOUN}
                unit={ENTITY_NOUN}
                options={neighborhoods}
                value={communityIds}
                onChange={setCommunityIds}
                testId="facet-neighborhoods"
              />
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="grid content-start gap-2 text-sm">
                  <span className="text-sm font-medium leading-none">Minimum confidence ({confidence.toFixed(2)})</span>
                  <Slider
                    aria-label="Minimum confidence"
                    value={[confidence]}
                    max={1}
                    step={0.05}
                    onValueChange={(value) => setConfidence(value[0] ?? 0)}
                  />
                </label>
                <div className="grid content-start gap-2">
                  <label className="flex items-center gap-2 text-sm">
                    <Switch checked={includeIsolates} onCheckedChange={setIncludeIsolates} aria-label="Include isolated entities" />
                    Include isolated entities
                  </label>
                  <label
                    className="flex items-center gap-2 text-sm"
                    title="Relations every source only claims, proposes or allows - a stated use, a hypothesis - rather than reports"
                  >
                    <Switch checked={includeClaimed} onCheckedChange={setIncludeClaimed} aria-label="Include claimed relations" />
                    Include claimed relations
                    <span className="text-muted-foreground" data-testid="claimed-count">({claimedCount.toLocaleString()})</span>
                  </label>
                </div>
              </div>
            </div>
            {analyst ? null : (
              // A perspective is this search and these neighborhoods with a
              // name, so it is saved from where they are set.
              <div className="flex flex-wrap items-center gap-2 border-t border-border bg-muted/30 px-3 py-2" data-testid="perspective-save">
                <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Perspective</span>
                {perspectiveId ? (
                  <>
                    <span className="text-sm">{namedPerspectives.find((item) => item.id === perspectiveId)?.name ?? "Selected"}</span>
                    <Button
                      size="sm"
                      variant="secondary"
                      className="h-7"
                      disabled={goldPending || !goldPresent || missingGold.length > 0}
                      onClick={() => {
                        if (goldPending) {
                          toast.error("Quality is still loading. Wait for gold before publishing.");
                          return;
                        }
                        if (!goldPresent) {
                          toast.error("No gold dataset is attached to this graph. Publish after gold is green.");
                          return;
                        }
                        if (missingGold.length > 0) {
                          toast.error(`Gold still missing ${missingGold.length} required relations. Publish after gold.`);
                          return;
                        }
                        promotePerspective.mutate({ id: perspectiveId, lifecycle: "production" });
                      }}
                      pending={promotePerspective.isPending}
                    >
                      {promotePerspective.isPending ? "Publishing…" : "Publish perspective"}
                    </Button>
                  </>
                ) : null}
                <Input
                  aria-label="Perspective name"
                  className="h-7 w-44 bg-background"
                  value={perspectiveName}
                  onChange={(event) => setPerspectiveName(event.target.value)}
                  placeholder="Name this view"
                />
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7"
                  pending={savePerspective.isPending}
                  disabled={!perspectiveName.trim()}
                  title={perspectiveName.trim() ? undefined : "Name the perspective before saving it"}
                  onClick={() =>
                    savePerspective.mutate(
                      {
                        graphId: dataset.graphId,
                        name: perspectiveName.trim(),
                        lifecycle: "draft",
                        search,
                        communityIds,
                        suggestedQuestions: search ? [`What connects to ${search}?`] : [],
                      },
                      { onSuccess: () => setPerspectiveName("") },
                    )
                  }
                >
                  {savePerspective.isPending ? "Saving…" : "Save as perspective"}
                </Button>
              </div>
            )}
          </PopoverContent>
        </Popover>
        {filtersActive ? (
          <Button size="sm" variant="ghost" className="h-8 px-2 text-xs" aria-label="Clear filters" onClick={clearFilters}>
            Clear
          </Button>
        ) : null}
        {namedPerspectives.length > 0 ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="outline" className="h-8 gap-1">
                {namedPerspectives.find((item) => item.id === perspectiveId)?.name ?? "Perspectives"}
                <ChevronDown className="size-3.5" aria-hidden="true" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              {namedPerspectives.map((item) => (
                <DropdownMenuItem
                  key={item.id}
                  onSelect={() => {
                    setPerspectiveId(item.id);
                    setSearch(item.search);
                    setCommunityIds(item.communityIds);
                  }}
                >
                  <span>{item.name} · {item.lifecycle}</span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
        {canvas.nodes.length > 0 ? (
          <>
            <span className="hidden h-6 w-px bg-border sm:block" aria-hidden="true" />
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span>Color</span>
              <Tabs value={colorMode} onValueChange={(value) => setColorMode(value as ColorMode)}>
                <TabsList className="h-7">
                  <TabsTrigger value="type" className="h-6 px-2 text-xs">
                    Type
                  </TabsTrigger>
                  <TabsTrigger value="community" className="h-6 px-2 text-xs" disabled={neighborhoods.length === 0}>
                    Neighborhood
                  </TabsTrigger>
                  <TabsTrigger value="degree" className="h-6 px-2 text-xs">
                    Degree
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span>Layout</span>
              <Tabs value={layoutMode} onValueChange={(value) => setLayoutMode(value as LayoutMode)}>
                <TabsList className="h-7">
                  <TabsTrigger value="force" className="h-6 px-2 text-xs">
                    Force
                  </TabsTrigger>
                  <TabsTrigger value="clustered" className="h-6 px-2 text-xs" disabled={neighborhoods.length === 0}>
                    Clustered
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span>Focus</span>
              <Tabs
                value={focusDepth === null ? "off" : String(focusDepth)}
                onValueChange={(value) => {
                  if (value === "off") {
                    setFocusDepth(null);
                    return;
                  }
                  const depth = Number(value);
                  if (depth === 1 || depth === 2 || depth === 3) {
                    setFocusDepth(depth);
                  }
                }}
              >
                <TabsList className="h-7">
                  <TabsTrigger value="off" className="h-6 px-2 text-xs">
                    Focus off
                  </TabsTrigger>
                  <TabsTrigger value="1" className="h-6 px-2 text-xs" disabled={!selectedId}>
                    1 hop
                  </TabsTrigger>
                  <TabsTrigger value="2" className="h-6 px-2 text-xs" disabled={!selectedId}>
                    2 hops
                  </TabsTrigger>
                  <TabsTrigger value="3" className="h-6 px-2 text-xs" disabled={!selectedId}>
                    3 hops
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </>
        ) : null}
        <p className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground" data-testid="explore-scope" title={scopeExplanation}>
          <span>{scopeStat}</span>
          <span
            role="img"
            aria-label="Scroll to zoom, drag to pan, double-click an entity to fly in. Shift-click two entities to find a path. A dashed relation is one its sources only claim."
            title="Scroll to zoom, drag to pan, double-click an entity to fly in. Shift-click two entities to find a path. A dashed relation is one its sources only claim."
          >
            <HelpCircle className="size-3.5" aria-hidden="true" />
          </span>
        </p>
      </div>
      <div className={cn("grid items-start gap-4", selectedNode || selectedEdge ? "lg:grid-cols-[2fr_1fr]" : "")}>
        <div className="relative">
          <GraphCanvas
            nodes={canvas.nodes}
            edges={canvas.edges}
            selectedId={selectedId}
            onSelect={setSelectedId}
            compact={canvas.nodes.length === 0}
            colorMode={colorMode}
            layoutMode={layoutMode}
            focusDepth={showNeighborhood ? 1 : focusDepth}
            pathHighlight={pathHighlight}
            hiddenTypes={hiddenTypeSet}
            membership={membership}
            flashNonce={flashNonce}
            showMinimap={showMinimap}
            onToggleMinimap={() => setShowMinimap((current) => !current)}
            onHover={setHover}
            onShiftClickNode={onShiftClickNode}
          />
          {canvas.nodes.length === 0 ? (
            <p className="pointer-events-none absolute inset-0 flex items-center justify-center p-6 text-center text-sm font-medium text-foreground">
              No entities to draw.{filtersActive ? " Clear filters to return to the overview." : ""}
            </p>
          ) : null}
          {pathEndpoints.length > 0 ? (
            <div className="absolute top-2 left-2 z-10 max-w-xs rounded-md border border-border bg-background/95 px-2 py-1.5 text-[11px] shadow-sm backdrop-blur-sm">
              <p className="font-semibold">Path finder</p>
              <p>
                {pathNames[0] ?? "—"}
                {pathEndpoints.length > 1 ? ` → ${pathNames[1]}` : " → shift-click another entity"}
              </p>
              {pathEndpoints.length === 2 ? (
                <p className="text-muted-foreground" role="status">
                  {pathHighlight && pathHighlight.edgeIds.size > 0
                    ? `${pathHighlight.edgeIds.size} hop${pathHighlight.edgeIds.size === 1 ? "" : "s"}`
                    : "No path found"}
                </p>
              ) : null}
              <button type="button" className="text-primary hover:underline" onClick={() => setPathEndpoints([])}>
                Clear path
              </button>
            </div>
          ) : null}
          {typeLegend.length > 0 && colorMode === "type" ? (
            <CanvasLegend
              key="types"
              title="Entity types"
              listLabel="Entity type colors"
              items={typeLegend}
              colorOf={colorForType}
              hiddenIds={hiddenTypeSet}
              onToggle={toggleHiddenType}
              onShowAll={() => setHiddenTypes([])}
            />
          ) : colorMode === "community" && neighborhoods.length > 0 ? (
            <CanvasLegend
              key="neighborhoods"
              title="Neighborhoods"
              listLabel="Neighborhood colors"
              items={neighborhoods}
              colorOf={colorForCommunity}
            />
          ) : null}
        </div>
        {selectedNode || selectedEdge ? (
        <Card data-testid="explore-selection">
          <CardHeader>
            <CardTitle>Selection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {selectedNode ? (
              <>
                <p className="font-medium">{String(selectedNode.name ?? selectedNode.id)}</p>
                <p className="text-muted-foreground">{String(selectedNode.primary_type ?? selectedNode.type ?? "")}</p>
                <p>{String(selectedNode.description ?? "")}</p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant={showNeighborhood ? "default" : "outline"}
                    aria-pressed={showNeighborhood}
                    onClick={() => setShowNeighborhood((current) => !current)}
                  >
                    {showNeighborhood ? "Hide neighborhood" : "Show neighborhood"}
                  </Button>
                  {onPin ? (
                    <Button size="sm" variant="outline" onClick={() => onPin(String(selectedNode.id), "node")}>
                      Looks wrong
                    </Button>
                  ) : null}
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={!dataset.evidence.some((row) => evidenceEntityId(row) === String(selectedNode.id) && String(row.quote ?? "").trim())}
                    onClick={() => {
                      const evidence = dataset.evidence.find(
                        (row) => evidenceEntityId(row) === String(selectedNode.id) && String(row.quote ?? "").trim(),
                      );
                      if (!evidence?.quote) return;
                      const quote = String(evidence.quote);
                      const documentId = evidence ? documentNames.get(evidenceDocumentId(evidence)) ?? evidenceDocumentId(evidence) : "";
                      void navigator.clipboard.writeText(`${documentId} · ${quote}`.trim());
                      toast.success("Copied citation");
                    }}
                  >
                    Copy citation
                  </Button>
                </div>
              </>
            ) : selectedEdge ? (
              <>
                <p className="flex items-center gap-2 font-medium">
                  {String(selectedEdge.relation_type)}
                  {isClaimed(selectedEdge) ? (
                    <Badge variant="warning" data-testid="edge-claimed">
                      Claimed
                    </Badge>
                  ) : null}
                </p>
                <p>
                  {String(selectedEdge.source_node_id ?? selectedEdge.source)} →{" "}
                  {String(selectedEdge.target_node_id ?? selectedEdge.target)}
                </p>
                {selectedEdge.description ? (
                  <p className="text-muted-foreground">{String(selectedEdge.description)}</p>
                ) : null}
                {isClaimed(selectedEdge) ? (
                  <p className="text-muted-foreground">
                    Every source only claims, proposes or allows this relation - a stated use or a hypothesis - and none
                    reports it as a finding.
                  </p>
                ) : null}
                {statedTypes(selectedEdge).length > 0 ? (
                  <p data-testid="edge-stated-types" className="text-muted-foreground">
                    Stated as {statedTypes(selectedEdge).join(", ")}; the ontology does not allow that type between these
                    entity types, so it is kept as {String(selectedEdge.relation_type)}.
                  </p>
                ) : null}
                {quantityLine(selectedEdge) ? (
                  <p data-testid="edge-quantities">
                    <span className="text-muted-foreground">Values: </span>
                    {quantityLine(selectedEdge)}
                  </p>
                ) : null}
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant={showNeighborhood ? "default" : "outline"}
                    aria-pressed={showNeighborhood}
                    onClick={() => setShowNeighborhood((current) => !current)}
                  >
                    {showNeighborhood ? "Hide neighborhood" : "Show neighborhood"}
                  </Button>
                  {onPin ? (
                    <Button size="sm" variant="outline" onClick={() => onPin(String(selectedEdge.id), "edge")}>
                      Looks wrong
                    </Button>
                  ) : null}
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={!dataset.evidence.some((row) => evidenceRelationId(row) === String(selectedEdge.id) && String(row.quote ?? "").trim())}
                    onClick={() => {
                      const evidence = dataset.evidence.find(
                        (row) => evidenceRelationId(row) === String(selectedEdge.id) && String(row.quote ?? "").trim(),
                      );
                      if (!evidence?.quote) return;
                      const quote = String(evidence.quote);
                      const documentId = evidence ? documentNames.get(evidenceDocumentId(evidence)) ?? evidenceDocumentId(evidence) : "";
                      void navigator.clipboard.writeText(`${documentId} · ${quote}`.trim());
                      toast.success("Copied citation");
                    }}
                  >
                    Copy citation
                  </Button>
                </div>
              </>
            ) : null}
          </CardContent>
        </Card>
        ) : null}
      </div>
      {hover && (hoverNode || hoverEdge) ? (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded-md border border-border bg-background px-2 py-1.5 text-[11px] shadow-md"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
          role="tooltip"
        >
          {hoverNode ? (
            <>
              <p className="font-semibold">{graphNodeLabel(hoverNode)}</p>
              <p className="text-muted-foreground">{String(hoverNode.primary_type ?? hoverNode.type ?? "")}</p>
              {hoverNode.description ? (
                <p className="mt-0.5 line-clamp-3 text-muted-foreground">{String(hoverNode.description)}</p>
              ) : null}
            </>
          ) : hoverEdge ? (
            <>
              <p className="font-semibold">
                {String(hoverEdge.relation_type ?? "")}
                {isClaimed(hoverEdge) ? <span className="ml-1 font-normal text-muted-foreground">(claimed)</span> : null}
              </p>
              <p className="text-muted-foreground">
                {String(hoverEdge.source_node_id ?? hoverEdge.source)} → {String(hoverEdge.target_node_id ?? hoverEdge.target)}
              </p>
              {hoverEdge.description ? (
                <p className="mt-0.5 line-clamp-3 text-muted-foreground">{String(hoverEdge.description)}</p>
              ) : null}
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function CanvasLegend({
  title,
  listLabel,
  items,
  colorOf,
  hiddenIds,
  onToggle,
  onShowAll,
}: {
  title: string;
  listLabel: string;
  /** Ranked largest first; only the first LEGEND_LIMIT show until expanded. */
  items: readonly FacetOption[];
  colorOf: (id: string) => string;
  /** With a toggle, each swatch is a button that hides or shows its type. */
  hiddenIds?: ReadonlySet<string>;
  onToggle?: (id: string) => void;
  onShowAll?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? items : items.slice(0, LEGEND_LIMIT);
  const folded = items.slice(shown.length);
  return (
    <div className="absolute bottom-2 left-2 z-10 max-w-[min(100%,28rem)] rounded-md border border-border bg-background/95 px-2 py-1.5 shadow-sm backdrop-blur-sm">
      <div className="mb-1 flex items-center justify-between gap-3">
        <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
        {hiddenIds && hiddenIds.size > 0 && onShowAll ? (
          <button type="button" className="text-[10px] font-medium text-primary hover:underline" onClick={onShowAll}>
            Show all
          </button>
        ) : null}
      </div>
      <ul className="flex flex-wrap items-center gap-1.5 text-[11px]" aria-label={listLabel}>
        {shown.map((item) => {
          const swatch = (
            <span className="size-2.5 shrink-0 rounded-full" style={{ background: colorOf(item.id) }} aria-hidden="true" />
          );
          if (!onToggle) {
            return (
              <li key={item.id} className="inline-flex items-center gap-1.5 px-1.5 py-0.5">
                {swatch}
                {item.label}
              </li>
            );
          }
          const hidden = hiddenIds?.has(item.id) ?? false;
          return (
            <li key={item.id}>
              <button
                type="button"
                aria-pressed={!hidden}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-sm px-1.5 py-0.5",
                  hidden ? "text-muted-foreground line-through opacity-50" : "text-foreground",
                )}
                onClick={() => onToggle(item.id)}
              >
                {swatch}
                {item.label}
                <span className="text-muted-foreground">{item.count.toLocaleString()}</span>
              </button>
            </li>
          );
        })}
        {folded.length > 0 ? (
          <li>
            <button
              type="button"
              className="rounded-sm px-1.5 py-0.5 text-muted-foreground hover:text-foreground hover:underline"
              title={folded.map((item) => item.label).join(", ")}
              aria-expanded={false}
              onClick={() => setExpanded(true)}
            >
              +{folded.length.toLocaleString()} more
            </button>
          </li>
        ) : items.length > LEGEND_LIMIT ? (
          <li>
            <button
              type="button"
              className="rounded-sm px-1.5 py-0.5 text-muted-foreground hover:text-foreground hover:underline"
              aria-expanded
              onClick={() => setExpanded(false)}
            >
              Show fewer
            </button>
          </li>
        ) : null}
      </ul>
    </div>
  );
}

/**
 * The four record tabs under the canvas. Entities and relations are the
 * filtered graph itself; communities and evidence follow it, so a filter
 * that hides an entity also hides the communities it alone populated and
 * the quotes that only ground it. Tables never see more than the graph
 * loaded, so "total" is the loaded count, not the summary metric.
 */
