"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Lock } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ShowMore } from "@/components/console/show-more";
import {
  DEFAULT_FOLDER_RELATION,
  type FolderEntitySelection,
  type OntologySelection,
} from "@/server/protocol/schema";
import {
  folderCounts,
  folderEntityProblem,
  folderPreview,
  levelExamples,
  MAX_FOLDER_LEVEL,
  pathParts,
  sharedFolderDepth,
  suggestedFolderLevel,
  uploadedPath,
} from "@/lib/folder-entity";
import { normalizeTermName } from "@/lib/ontology-terms";
import { cn } from "@/lib/utils";

/** Files listed in the preview before "Show more". */
export const FOLDER_PREVIEW_PAGE_SIZE = 50;

const NO_FOLDER: FolderEntitySelection = {
  entityType: "",
  relationType: DEFAULT_FOLDER_RELATION,
  level: 1,
  excludedFileIds: [],
  excludedPaths: [],
};

/**
 * Tie every document to one entity its folder names - a study, a contract -
 * and link everything the document states to it. Which folder is read is a
 * level below the upload root; the preview shows the folder each uploaded
 * file is tied to, and a file can be left out of its folder here.
 *
 * Until someone picks a level, it follows the uploads: below any folder every
 * file shares, as a picked or dropped folder does, to the first folder that
 * tells the files apart.
 */
export function FolderEntityCard({
  ontology,
  onChange,
  paths,
  locked = null,
}: {
  ontology: OntologySelection;
  onChange: (folder: FolderEntitySelection | null) => void;
  /** The uploaded files' paths below the upload root. */
  paths: readonly string[];
  /** When set, the setting is shown but not editable, for this reason. */
  locked?: string | null;
}) {
  const folder = ontology.folderEntity ?? null;
  const [limit, setLimit] = useState(FOLDER_PREVIEW_PAGE_SIZE);
  const [levelChosen, setLevelChosen] = useState(false);
  // What is typed in the relation field, until it loses focus.
  const [relationDraft, setRelationDraft] = useState<string | null>(null);
  const suggested = useMemo(() => suggestedFolderLevel(paths), [paths]);
  const examples = useMemo(() => levelExamples(paths), [paths]);
  const shared = useMemo(() => pathParts(paths[0] ?? "").slice(0, sharedFolderDepth(paths)).join("/"), [paths]);
  const following = folder !== null && locked === null && !levelChosen && paths.length > 0;
  useEffect(() => {
    if (following && folder.level !== suggested) {
      onChange({ ...folder, level: suggested });
    }
  }, [following, folder, suggested, onChange]);
  const rows = useMemo(() => (folder ? folderPreview(paths, folder) : []), [folder, paths]);
  const folders = useMemo(() => folderCounts(rows), [rows]);
  const problem = folderEntityProblem(ontology);
  const change = (update: Partial<FolderEntitySelection>) => onChange({ ...(folder ?? NO_FOLDER), ...update });
  const toggle = (path: string) => {
    if (!folder) return;
    const excluded = new Set(folder.excludedPaths.map(uploadedPath));
    if (excluded.has(path)) {
      excluded.delete(path);
    } else {
      excluded.add(path);
    }
    change({ excludedPaths: [...excluded].sort() });
  };
  const page = rows.slice(0, limit);
  return (
    <Card data-testid="folder-entity">
      <CardHeader>
        <CardTitle>Folders</CardTitle>
        <CardDescription>
          When the folder a file is filed in says what it belongs to, tie every document to one entity named by its
          folder and link everything the document states to it. Those links are kept out of community detection, so
          communities still follow topics.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <label className="flex items-center gap-3 text-sm font-medium">
          <Switch
            checked={folder !== null}
            disabled={locked !== null}
            onCheckedChange={(checked) =>
              onChange(checked ? { ...NO_FOLDER, level: levelChosen ? NO_FOLDER.level : suggested } : null)
            }
            aria-label="Tie documents to their folder"
          />
          Tie documents to their folder
        </label>
        {locked ? (
          <p className="flex items-start gap-2 text-xs text-muted-foreground">
            <Lock className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            {locked}
          </p>
        ) : null}
        {folder ? (
          <>
            <div className="grid gap-4 sm:grid-cols-3">
              <FolderField label="Folder entity type">
                <Select
                  value={folder.entityType || undefined}
                  disabled={locked !== null}
                  onValueChange={(value) => change({ entityType: value })}
                >
                  <SelectTrigger aria-label="Folder entity type">
                    <SelectValue placeholder="Choose an entity type" />
                  </SelectTrigger>
                  <SelectContent>
                    {ontology.entityTypes.map((term) => (
                      <SelectItem key={term.name} value={term.name}>
                        {term.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FolderField>
              <FolderField label="Relation">
                <Input
                  aria-label="Folder relation"
                  placeholder={DEFAULT_FOLDER_RELATION}
                  disabled={locked !== null}
                  value={relationDraft ?? (folder.relationType === DEFAULT_FOLDER_RELATION ? "" : folder.relationType)}
                  onChange={(event) => setRelationDraft(event.target.value)}
                  onBlur={() => {
                    if (relationDraft !== null) {
                      change({ relationType: folderRelationName(relationDraft) });
                      setRelationDraft(null);
                    }
                  }}
                />
              </FolderField>
              <FolderField label="Folder level">
                <Select
                  value={String(folder.level)}
                  disabled={locked !== null}
                  onValueChange={(value) => {
                    setLevelChosen(true);
                    change({ level: Number(value) });
                  }}
                >
                  <SelectTrigger aria-label="Folder level">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: MAX_FOLDER_LEVEL }, (_, index) => index + 1).map((level) => (
                      <SelectItem key={level} value={String(level)}>
                        {levelLabel(level, examples.get(level), paths.length > 0)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FolderField>
            </div>
            <p className="text-xs text-muted-foreground">
              The entity is named by the folder&apos;s own name at this level, and folders of one name at this level
              are one entity: choose the level whose folder names say what the files belong to.
              {shared && folder.level > 1
                ? ` Every file is in ${shared}, so the folders that tell them apart start at level ${pathParts(shared).length + 1}.`
                : ""}
            </p>
            {problem ? <p className="text-sm text-destructive">{problem}</p> : null}
            {rows.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                Upload a folder to see the folder each file is tied to. A file with no folder at this level is tied to
                none.
              </p>
            ) : (
              <div className="space-y-3" data-testid="folder-preview">
                <p className="text-sm text-muted-foreground" data-testid="folder-preview-summary">
                  {folders.length.toLocaleString()} folder{folders.length === 1 ? "" : "s"} at this level
                  {folders.length ? `: ${folders.map((item) => `${item.folder} (${item.files})`).join(", ")}` : ""}
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-16">Tie</TableHead>
                      <TableHead>File</TableHead>
                      <TableHead>Folder</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {page.map((row) => (
                      <TableRow key={row.path} data-excluded={row.excluded ? "true" : undefined}>
                        <TableCell>
                          <input
                            type="checkbox"
                            aria-label={`Tie ${row.path} to its folder`}
                            checked={row.folder !== null && !row.excluded}
                            disabled={row.folder === null || locked !== null}
                            onChange={() => toggle(row.path)}
                          />
                        </TableCell>
                        <TableCell className={cn("max-w-xs truncate", row.excluded && "text-muted-foreground line-through")} title={row.path}>
                          {row.path}
                        </TableCell>
                        <TableCell className="max-w-xs truncate text-muted-foreground" title={row.folder ?? undefined}>
                          {row.name ?? "No folder at this level"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <ShowMore
                  shown={page.length}
                  total={rows.length}
                  pageSize={FOLDER_PREVIEW_PAGE_SIZE}
                  noun="files"
                  onMore={() => setLimit((current) => current + FOLDER_PREVIEW_PAGE_SIZE)}
                />
              </div>
            )}
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}

function FolderField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
    </label>
  );
}

/** A level option, named by a folder the uploads hold at that level. */
function levelLabel(level: number, example: string | undefined, uploaded: boolean): string {
  if (example) {
    return `${level} · e.g. ${example}`;
  }
  if (uploaded) {
    return `${level} · no file is this deep`;
  }
  return level === 1 ? "1 · the top folder" : `${level} · ${level} folders down`;
}

/**
 * The folder relation for what was typed: a relation type's name, the default
 * when nothing was, and as typed when it cannot be one, which the card then
 * says.
 */
export function folderRelationName(typed: string): string {
  const raw = typed.trim();
  return raw ? (normalizeTermName(raw) ?? raw) : DEFAULT_FOLDER_RELATION;
}
