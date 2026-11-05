/**
 * How an evidence or chunk row names what it grounds and where it came from.
 *
 * The pipeline's parquet says `subject_id` with a `subject_kind` of node or
 * edge and a `file_id`; the gold format and flat exports say `entity_id`,
 * `relation_id` and `document_id`. Every reader goes through here.
 */
export function evidenceEntityId(row: Record<string, unknown>): string | null {
  const explicit = row.entity_id ?? row.node_id;
  if (explicit != null && String(explicit)) {
    return String(explicit);
  }
  if (String(row.subject_kind ?? "") === "node" && row.subject_id != null) {
    return String(row.subject_id);
  }
  return null;
}

export function evidenceRelationId(row: Record<string, unknown>): string | null {
  const explicit = row.relation_id ?? row.edge_id;
  if (explicit != null && String(explicit)) {
    return String(explicit);
  }
  if (String(row.subject_kind ?? "") === "edge" && row.subject_id != null) {
    return String(row.subject_id);
  }
  return null;
}

export function evidenceDocumentId(row: Record<string, unknown>): string {
  const value = row.document_id ?? row.file_id;
  return value == null ? "" : String(value);
}

/** Document id → the name a reader knows it by: its filename, else its source, else the id. */
export function documentNameIndex(documents: readonly Record<string, unknown>[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const document of documents) {
    const name = documentDisplayName(document);
    for (const key of [document.id, document.file_id, document.document_id]) {
      if (key != null && String(key) && name) {
        names.set(String(key), name);
      }
    }
  }
  return names;
}

/**
 * The values a relation states - limits, ranges, results - as one readable line,
 * e.g. "limit: ≤ 0.5 % · result: 0.1 %". Empty when the relation states none.
 */
export function quantityLine(edge: Record<string, unknown>): string {
  const quantities = Array.isArray(edge.quantities) ? edge.quantities : [];
  return quantities
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map((item) => {
      const text = String(item.text ?? "");
      const unit = item.unit ? String(item.unit) : "";
      const shown = unit && !text.includes(unit) ? `${text} ${unit}` : text;
      return item.kind ? `${String(item.kind)}: ${shown}` : shown;
    })
    .filter(Boolean)
    .join(" · ");
}

const URI_SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i;

function decodedSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

export function documentDisplayName(document: Record<string, unknown>): string {
  const filename = document.filename ?? document.name ?? document.title;
  if (filename != null && String(filename)) {
    return String(filename);
  }
  const uri = document.source_uri ?? document.path ?? document.uri;
  if (uri != null && String(uri)) {
    const text = String(uri);
    const tail = text.split(/[\\/]/).filter(Boolean).pop();
    if (!tail) {
      return text;
    }
    // A URI escapes its path ('%20' for a space); a plain path's '%' is literal.
    return URI_SCHEME.test(text) ? decodedSegment(tail) : tail;
  }
  return String(document.id ?? document.file_id ?? "");
}

/**
 * Entity id → the document that grounds it: the file of its first evidence
 * quote, else the file of the first chunk it was extracted from.
 */
export function entityDocumentIndex(dataset: {
  nodes: readonly Record<string, unknown>[];
  evidence: readonly Record<string, unknown>[];
  chunks: readonly Record<string, unknown>[];
}): Map<string, string> {
  const documents = new Map<string, string>();
  for (const row of dataset.evidence) {
    const entityId = evidenceEntityId(row);
    const documentId = evidenceDocumentId(row);
    if (entityId && documentId && !documents.has(entityId)) {
      documents.set(entityId, documentId);
    }
  }
  const chunkDocument = new Map<string, string>();
  for (const chunk of dataset.chunks) {
    const id = chunk.id == null ? "" : String(chunk.id);
    const documentId = evidenceDocumentId(chunk);
    if (id && documentId) {
      chunkDocument.set(id, documentId);
    }
  }
  for (const node of dataset.nodes) {
    const id = node.id == null ? "" : String(node.id);
    if (!id || documents.has(id)) {
      continue;
    }
    const chunkIds = Array.isArray(node.source_chunk_ids) ? node.source_chunk_ids : [];
    for (const chunkId of chunkIds) {
      const documentId = chunkDocument.get(String(chunkId));
      if (documentId) {
        documents.set(id, documentId);
        break;
      }
    }
  }
  return documents;
}

/** The relation types the model stated for an edge the ontology's fallback relation kept. */
export function statedTypes(edge: Record<string, unknown>): string[] {
  const stated = edge.stated_relation_types;
  return Array.isArray(stated) ? stated.map(String).filter(Boolean) : [];
}

/**
 * Whether every source only claims an edge - a stated use, a proposal, a
 * hypothesis - rather than reporting it as a fact or finding. One reporting
 * source makes the edge reported.
 */
export function isClaimed(row: Record<string, unknown>): boolean {
  return row.assertion === "claimed";
}

const EVIDENCE_METHOD_LABELS: Record<string, string> = {
  llm: "Quoted by the model",
  llm_located: "Lines the model pointed at",
  llm_confirmed: "Model-confirmed, in other words",
  ocr_tolerant: "Matched despite OCR errors",
  llm_verified: "Lines the model pointed at, verified with the document's opening",
  ontology_cue: "Ontology cue",
  type_resolved: "Admitted on types other documents give",
  folder: "The folder the document is filed in",
};

/**
 * How a piece of evidence was found, readably. "Model-confirmed" evidence is
 * exact source text that states its subject in other words, and OCR-tolerant
 * evidence names it only as OCR printed it - words run together or a letter
 * misread - so a reader who wants only verbatim names can tell both apart. "Admitted on types" marks a
 * relation its window rejected for its entity types and finalization admitted
 * because other documents type those entities as the relation allows. Folder
 * evidence is the folder line of a document's location page: the entity its
 * folder names, and the links from what the document states to it.
 */
export function evidenceMethodLabel(method: unknown): string {
  const key = typeof method === "string" && method ? method : "llm";
  return EVIDENCE_METHOD_LABELS[key] ?? key;
}
