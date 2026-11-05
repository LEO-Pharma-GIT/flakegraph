import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { ontologySelectionFromProfile } from "./ontology-profile";
import type { OntologySelection } from "./protocol/schema";

/** A hosted corpus the console can point a first run at, when it is on this host. */
export interface SamplePack {
  name: string;
  /** Relative to the repository root, as the pipeline's `files.input_path`. */
  path: string;
  graphName: string;
  why: string;
  /**
   * The vocabulary the pack's gold is written in (its `ontology.yaml`), so a
   * run on the pack extracts what its benchmark scores. Null without one.
   */
  ontology: OntologySelection | null;
}

const SAMPLE_PACKS: ReadonlyArray<Omit<SamplePack, "ontology">> = [
  {
    name: "Martial arts",
    path: "data/martial_arts/files",
    graphName: "Martial arts history",
    why: "10 markdown files, gold.json beside them",
  },
  {
    name: "Atopic dermatitis",
    path: "data/atopic_dermatitis/files",
    graphName: "Atopic dermatitis treatment",
    why: "50 open-access papers on its treatment, gold.json beside them",
  },
  {
    name: "Deep learning papers",
    path: "data/deep_learning_papers/files",
    graphName: "Deep learning papers",
    why: "PDF-heavy sample with gold relations",
  },
];

/**
 * The packs whose files are actually here.
 *
 * A laptop checkout has both; the fleet image ships the small one and the
 * papers must be downloaded first. Offering a pack that is not on the host
 * only produces "Input path does not exist" after the click.
 */
export async function availableSamplePacks(repositoryRoot: string): Promise<SamplePack[]> {
  const packs = await Promise.all(
    SAMPLE_PACKS.map(async (pack) => {
      try {
        if (!(await stat(path.join(repositoryRoot, pack.path))).isDirectory()) {
          return null;
        }
      } catch {
        return null;
      }
      const profile = await samplePackProfile(repositoryRoot, pack.path);
      return { ...pack, ontology: profile ? ontologySelectionFromProfile(profile) : null };
    }),
  );
  return packs.filter((pack): pack is SamplePack => pack !== null);
}

/**
 * The ontology profile a sample pack ships beside its files, when the source
 * is one of the packs; null for any other folder. Only the known packs are
 * read, so a source path cannot point this at an arbitrary file.
 */
export async function samplePackProfile(
  repositoryRoot: string,
  sourcePath: string,
): Promise<Record<string, unknown> | null> {
  // A request's path arrives resolved against the repository, as confined.
  const resolved = path.resolve(repositoryRoot, sourcePath);
  const pack = SAMPLE_PACKS.find((item) => path.resolve(repositoryRoot, item.path) === resolved);
  if (!pack) {
    return null;
  }
  try {
    const profile = parseYaml(await readFile(path.join(repositoryRoot, path.dirname(pack.path), "ontology.yaml"), "utf8"));
    return profile && typeof profile === "object" ? (profile as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}
