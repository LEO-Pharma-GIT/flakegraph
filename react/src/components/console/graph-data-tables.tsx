"use client";

import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RecordTable, type RecordColumn, type RecordRow } from "@/components/console/record-table";
import { communityMemberIds } from "@/server/graph-filter";
import type { GraphDataset } from "@/server/protocol/schema";
import {
  documentDisplayName,
  documentNameIndex,
  evidenceDocumentId,
  evidenceEntityId,
  evidenceRelationId,
  evidenceMethodLabel,
  isClaimed,
  quantityLine,
  statedTypes,
} from "@/lib/evidence";
import { colorForType, graphEdgeEnds, graphEdgeId, graphNodeId, graphNodeLabel } from "@/lib/graph-geometry";
import { formatProvenance } from "@/lib/utils";

/** The part of Explore's state the tables need: what its filters leave, and how to say so. */
export interface ExploreFocus {
  nodes: readonly RecordRow[];
  edges: readonly RecordRow[];
  /** "2 entity types · 1 neighborhood", or null when nothing narrows the graph. */
  summary: string | null;
}

/**
 * The graph as tables: entities, relations, neighborhoods and evidence.
 *
 * They read the same graph Explore draws, narrowed by the same filters, but
 * they are a different way of looking: a row is for scanning and sorting,
 * the canvas for seeing shape. So they get their own tab, and every row
 * offers the way back - "Show in graph" selects the record on the canvas,
 * or focuses a neighborhood there.
 */
export function GraphDataTables({
  dataset,
  focus,
  missingGold = [],
  selectedId,
  onShowInGraph,
  onFocusNeighborhood,
  onClearFilters,
}: {
  dataset: GraphDataset;
  focus: ExploreFocus;
  missingGold?: ReadonlyArray<{ id: string }>;
  selectedId: string | null;
  onShowInGraph: (id: string) => void;
  onFocusNeighborhood: (communityId: string) => void;
  onClearFilters: () => void;
}) {
  const documentNames = useMemo(() => documentNameIndex(dataset.documents ?? []), [dataset.documents]);
  const nodeNames = useMemo(
    () => new Map(dataset.nodes.map((node) => [graphNodeId(node), graphNodeLabel(node)] as const)),
    [dataset.nodes],
  );
  const edgesById = useMemo(
    () => new Map(dataset.edges.map((edge, index) => [graphEdgeId(edge, index), edge] as const)),
    [dataset.edges],
  );
  const visibleNodeIds = useMemo(() => new Set(focus.nodes.map((node) => graphNodeId(node))), [focus.nodes]);
  const visibleEdgeIds = useMemo(() => new Set(focus.edges.map((edge, index) => graphEdgeId(edge, index))), [focus.edges]);
  // Only a filter that removed something narrows the other two tabs; a row
  // whose subject the graph never had stays visible either way.
  const narrowed = focus.nodes.length < dataset.nodes.length || focus.edges.length < dataset.edges.length;
  const communities = useMemo(() => {
    if (!narrowed) {
      return dataset.communities;
    }
    return dataset.communities.filter((community) => {
      const members = communityMemberIds(community);
      return members.length === 0 || members.some((member) => visibleNodeIds.has(member));
    });
  }, [dataset.communities, narrowed, visibleNodeIds]);
  const evidence = useMemo(() => {
    if (!narrowed) {
      return dataset.evidence;
    }
    return dataset.evidence.filter((row) => {
      const relationId = evidenceRelationId(row);
      const entityId = evidenceEntityId(row);
      if (!relationId && !entityId) {
        return true;
      }
      return (relationId != null && visibleEdgeIds.has(relationId)) || (entityId != null && visibleNodeIds.has(entityId));
    });
  }, [dataset.evidence, narrowed, visibleEdgeIds, visibleNodeIds]);
  const highlightIds = useMemo(() => new Set(missingGold.map((item) => item.id)), [missingGold]);

  // Column definitions close over the name indexes, and the table re-sorts
  // whenever they change, so they are rebuilt only when an index does.
  const columns = useMemo(() => {
    const nameOf = (id: unknown) => (id == null ? "" : (nodeNames.get(String(id)) ?? String(id)));
    const relationLabel = (edge: RecordRow) => {
      const ends = graphEdgeEnds(edge);
      return `${nameOf(ends.source)} ${String(edge.relation_type ?? "")} ${nameOf(ends.target)}`.trim();
    };
    const entityColumns: RecordColumn[] = [
      { key: "name", value: (row) => graphNodeLabel(row) },
      { key: "primary_type", label: "Type", kind: "type", value: (row) => row.primary_type ?? row.type ?? "", color: colorForType },
      { key: "description", kind: "long" },
      { key: "id", kind: "id" },
    ];
    const relationColumns: RecordColumn[] = [
      {
        key: "source_node_id",
        label: "Source",
        value: (row) => nameOf(graphEdgeEnds(row).source),
        title: (row) => String(graphEdgeEnds(row).source),
      },
      { key: "relation_type", label: "Relation", kind: "type", value: (row) => row.relation_type ?? "" },
      {
        key: "target_node_id",
        label: "Target",
        value: (row) => nameOf(graphEdgeEnds(row).target),
        title: (row) => String(graphEdgeEnds(row).target),
      },
      {
        key: "stated_relation_types",
        label: "Stated as",
        kind: "text",
        value: (row) => statedTypes(row).join(", "),
      },
      { key: "quantities", label: "Values", kind: "long", value: (row) => quantityLine(row) },
      {
        key: "assertion",
        label: "Assertion",
        kind: "type",
        value: (row) => (isClaimed(row) ? "claimed" : "reported"),
      },
      { key: "confidence", kind: "number" },
      { key: "id", kind: "id" },
    ];
    const documentColumns: RecordColumn[] = [
      { key: "source_uri", label: "Document", width: "w-64", value: (row) => documentDisplayName(row) },
      { key: "kind", kind: "type", value: (row) => row.kind ?? "document" },
      { key: "folder", width: "w-64", value: (row) => row.folder ?? "" },
      {
        key: "provenance",
        label: "Provenance",
        kind: "type",
        value: (row) => formatProvenance(typeof row.provenance === "string" ? row.provenance : "extracted"),
      },
      { key: "summary", kind: "long" },
      {
        key: "related_file_ids",
        label: "Related files",
        kind: "long",
        value: (row) =>
          (Array.isArray(row.related_file_ids) ? row.related_file_ids : [])
            .map((id) => documentNames.get(String(id)) ?? String(id))
            .join(", "),
      },
    ];
    const communityColumns: RecordColumn[] = [
      { key: "title", value: (row) => row.title ?? row.id ?? "" },
      { key: "size", kind: "number", value: (row) => communityMemberIds(row).length },
      { key: "summary", kind: "long" },
    ];
    const evidenceColumns: RecordColumn[] = [
      {
        key: "document_id",
        label: "Document",
        width: "w-56",
        value: (row) => documentNames.get(evidenceDocumentId(row)) ?? evidenceDocumentId(row),
        title: (row) => evidenceDocumentId(row) || undefined,
      },
      { key: "quote", kind: "long" },
      { key: "method", label: "How found", width: "w-48", value: (row) => evidenceMethodLabel(row.method) },
      {
        key: "supports",
        width: "w-72",
        value: (row) => {
          const relationId = evidenceRelationId(row);
          const edge = relationId ? edgesById.get(relationId) : undefined;
          if (edge) {
            return relationLabel(edge);
          }
          return relationId ?? nameOf(evidenceEntityId(row));
        },
        title: (row) => evidenceRelationId(row) ?? evidenceEntityId(row) ?? undefined,
      },
    ];
    return {
      entities: entityColumns,
      relations: relationColumns,
      communities: communityColumns,
      evidence: evidenceColumns,
      documents: documentColumns,
    };
  }, [documentNames, edgesById, nodeNames]);

  const showInGraph = { label: "Show in graph", onClick: onShowInGraph };

  return (
    <div className="space-y-3" data-testid="graph-data">
      {focus.summary ? (
        <p className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground" data-testid="graph-data-scope">
          <span>
            Narrowed by the Explore filters: <span className="text-foreground">{focus.summary}</span>.
          </span>
          <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" onClick={onClearFilters}>
            Clear filters
          </Button>
        </p>
      ) : null}
      <Tabs defaultValue="entities">
        <TabsList className="h-auto min-h-9 w-auto max-w-full flex-wrap justify-start">
          <TabsTrigger value="entities">Entities</TabsTrigger>
          <TabsTrigger value="relations">Relations</TabsTrigger>
          <TabsTrigger value="communities">Neighborhoods</TabsTrigger>
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="documents">Documents</TabsTrigger>
        </TabsList>
        <TabsContent value="entities">
          <RecordTable
            rows={focus.nodes}
            total={dataset.nodes.length}
            columns={columns.entities}
            noun={{ one: "entity", many: "entities" }}
            action={showInGraph}
            selectedId={selectedId}
          />
        </TabsContent>
        <TabsContent value="relations">
          <RecordTable
            rows={focus.edges}
            total={dataset.edges.length}
            columns={columns.relations}
            noun={{ one: "relation", many: "relations" }}
            action={showInGraph}
            selectedId={selectedId}
            highlightIds={highlightIds}
          />
        </TabsContent>
        <TabsContent value="communities">
          <RecordTable
            rows={communities}
            total={dataset.communities.length}
            columns={columns.communities}
            noun={{ one: "neighborhood", many: "neighborhoods" }}
            action={{ label: "Focus in graph", onClick: onFocusNeighborhood }}
          />
        </TabsContent>
        <TabsContent value="evidence">
          <RecordTable
            rows={evidence}
            total={dataset.evidence.length}
            columns={columns.evidence}
            noun={{ one: "evidence row", many: "evidence rows" }}
            action={showInGraph}
            selectedId={selectedId}
            selectionId={(row) => evidenceRelationId(row) ?? evidenceEntityId(row)}
          />
        </TabsContent>
        <TabsContent value="documents">
          <RecordTable
            rows={dataset.documents ?? []}
            total={(dataset.documents ?? []).length}
            columns={columns.documents}
            noun={{ one: "document", many: "documents" }}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
