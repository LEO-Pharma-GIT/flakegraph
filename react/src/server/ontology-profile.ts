import { DEFAULT_FOLDER_RELATION, type FolderEntitySelection, type OntologySelection, type OntologyTerm } from "./protocol/schema";

type Profile = Record<string, unknown>;

/** Read a pipeline profile (inline or a fleet's mounted one) back into form terms. */
export function ontologySelectionFromProfile(profile: Profile, graph: Profile = {}): OntologySelection {
  const terms = (value: unknown): OntologyTerm[] =>
    Array.isArray(value)
      ? value
          .map((item) =>
            typeof item === "string"
              ? { name: item, description: "" }
              : item && typeof item === "object"
                ? {
                    name: String((item as Profile).name ?? ""),
                    description: String((item as Profile).description ?? ""),
                  }
                : { name: "", description: "" },
          )
          .filter((item) => item.name)
      : [];
  const entityTypes = terms(profile.entity_types);
  // A structural relation is the folder entity's, not a term extraction is told about.
  const relationTypes = terms(
    Array.isArray(profile.relation_types)
      ? profile.relation_types.filter((item) => !(item && typeof item === "object" && (item as Profile).structural === true))
      : [],
  );
  const mode = String(profile.mode ?? "hybrid");
  const folderEntity = folderEntityFromProfile(profile.folder_entity);
  return {
    entityTypes: entityTypes.length ? entityTypes : terms(graph.entity_types),
    relationTypes,
    relations: mode === "closed" ? "fixed" : mode === "open" || relationTypes.length === 0 ? "open" : "guided",
    ...(folderEntity ? { folderEntity } : {}),
  };
}

function folderEntityFromProfile(value: unknown): FolderEntitySelection | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const folder = value as Profile;
  const strings = (items: unknown): string[] => (Array.isArray(items) ? items.map(String) : []);
  const level = Number(folder.level ?? 1);
  return {
    entityType: String(folder.entity_type ?? ""),
    relationType: String(folder.relation_type ?? DEFAULT_FOLDER_RELATION),
    level: Number.isInteger(level) && level > 0 ? level : 1,
    excludedFileIds: strings(folder.excluded_file_ids),
    excludedPaths: strings(folder.excluded_paths),
  };
}

/** The pipeline's folder entity section for what the form selected. */
function folderEntityProfile(folder: FolderEntitySelection): Profile {
  return {
    entity_type: folder.entityType,
    relation_type: folder.relationType.trim() || DEFAULT_FOLDER_RELATION,
    level: folder.level,
    excluded_file_ids: [...folder.excludedFileIds],
    excluded_paths: [...folder.excludedPaths],
  };
}

/** The pipeline's one rule for when two labels are the same label. */
export function normalizeLabel(value: string): string {
  return value.toLocaleLowerCase().replaceAll("-", " ").split(/\s+/).filter(Boolean).join("_");
}

/**
 * The pipeline's ontology profile for what the form selected.
 *
 * The form edits names and descriptions; the profile the terms came from
 * (`base`: a sample pack's, the fleet's, the default one) also carries each
 * term's aliases, examples, endpoint types, inverse and evidence cues, which
 * steer extraction as much as the names do. A term the form kept keeps them,
 * so a run on the console extracts with the same profile a run from the
 * command line does. What an edit removed is dropped from what refers to it,
 * because the pipeline refuses a profile that names a type it lacks.
 */
export function ontologyProfile(selection: OntologySelection, base: Profile | null = null): Profile {
  const baseTerms = (value: unknown) =>
    new Map(
      (Array.isArray(value) ? value : [])
        .filter((item): item is Profile => Boolean(item) && typeof item === "object" && typeof (item as Profile).name === "string")
        .map((item) => [normalizeLabel(String(item.name)), item]),
    );
  const baseEntities = baseTerms(base?.entity_types);
  const baseRelations = baseTerms(base?.relation_types);
  const term = (item: OntologyTerm, from: Map<string, Profile>): Profile => {
    const name = item.name.trim();
    return {
      ...(from.get(normalizeLabel(name)) ?? {}),
      name,
      description:
        item.description.trim() ||
        String(from.get(normalizeLabel(name))?.description ?? "").trim() ||
        `A source-grounded ${name.replaceAll("_", " ").toLowerCase()}.`,
    };
  };
  const entityTypes = selection.entityTypes.map((item) => term(item, baseEntities));
  const entityNames = new Set(entityTypes.map((item) => String(item.name)));
  const chosenRelations = selection.relations === "open" ? [] : selection.relationTypes.map((item) => term(item, baseRelations));
  const relationNames = new Set(chosenRelations.map((item) => String(item.name)));
  const labels = new Set(chosenRelations.map((item) => normalizeLabel(String(item.name))));
  const relationTypes = chosenRelations.map((relation) => {
    const kept: Profile = { ...relation };
    for (const side of ["source_types", "target_types"] as const) {
      if (Array.isArray(kept[side])) {
        const types = (kept[side] as unknown[]).map(String).filter((type) => entityNames.has(type));
        if (types.length) {
          kept[side] = types;
        } else {
          delete kept[side];
        }
      }
    }
    if (typeof kept.inverse === "string" && !relationNames.has(kept.inverse)) {
      delete kept.inverse;
    }
    if (Array.isArray(kept.aliases)) {
      kept.aliases = (kept.aliases as unknown[]).map(String).filter((alias) => {
        const label = normalizeLabel(alias);
        if (labels.has(label)) {
          return false;
        }
        labels.add(label);
        return true;
      });
    }
    return kept;
  });
  const mode = selection.relations === "fixed" ? "closed" : selection.relations === "open" ? "open" : "hybrid";
  const profile: Profile = {
    ...(base ?? { name: "console", description: "Chosen on the console for this graph." }),
    mode,
    entity_types: entityTypes,
    relation_types: relationTypes,
  };
  // The pipeline adds the folder relation itself, from the folder entity.
  if (selection.folderEntity) {
    profile.folder_entity = folderEntityProfile(selection.folderEntity);
  } else {
    delete profile.folder_entity;
  }
  if (typeof profile.fallback_relation === "string" && !relationNames.has(profile.fallback_relation)) {
    delete profile.fallback_relation;
  }
  if (Array.isArray(profile.relation_rewrites)) {
    // A rewrite may name a relation by any label the pipeline resolves it by.
    const relationLabels = new Set(
      relationTypes.flatMap((relation) =>
        [relation.name, ...(Array.isArray(relation.aliases) ? relation.aliases : [])].map((label) => normalizeLabel(String(label))),
      ),
    );
    profile.relation_rewrites = profile.relation_rewrites.filter((value) => {
      if (!value || typeof value !== "object") return false;
      const rewrite = value as Profile;
      if (!relationLabels.has(normalizeLabel(String(rewrite.relation))) || !relationLabels.has(normalizeLabel(String(rewrite.to)))) return false;
      return ["source_types", "target_types"].every((side) =>
        !Array.isArray(rewrite[side]) || (rewrite[side] as unknown[]).every((type) => entityNames.has(String(type))),
      );
    });
  }
  return profile;
}
