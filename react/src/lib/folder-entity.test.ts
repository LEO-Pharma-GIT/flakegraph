import { describe, expect, it } from "vitest";
import { folderRelationName } from "@/components/console/folder-entity-card";
import { safeUploadPath } from "@/server/confine";
import type { FolderEntitySelection, OntologySelection } from "@/server/protocol/schema";
import {
  folderAtLevel,
  folderCounts,
  folderEntityProblem,
  folderExcluded,
  folderName,
  folderPreview,
  levelExamples,
  sharedFolderDepth,
  suggestedFolderLevel,
  uploadedPath,
} from "./folder-entity";

const folder: FolderEntitySelection = {
  entityType: "STUDY",
  relationType: "APPEARS_IN",
  level: 1,
  excludedFileIds: ["file-x"],
  excludedPaths: ["Northwind Pricing Study/notes.txt"],
};

describe("the folder a file is tied to", () => {
  it("is the path at the chosen level below the upload root, never the file itself", () => {
    expect(folderAtLevel("Studies/Northwind Pricing Study/raw/prices.txt", 1)).toBe("Studies");
    expect(folderAtLevel("Studies/Northwind Pricing Study/raw/prices.txt", 2)).toBe("Studies/Northwind Pricing Study");
    expect(folderAtLevel("Studies/overview.pdf", 2)).toBeNull();
    expect(folderAtLevel("readme.txt", 1)).toBeNull();
  });

  it("reads a picked path as the upload stores it", () => {
    expect(uploadedPath("\\Contract ACME-0042\\./terms.pdf")).toBe("Contract ACME-0042/terms.pdf");
    expect(uploadedPath("/a/../b/c.pdf")).toBe("a/b/c.pdf");
  });

  it("previews every picked file with its folder and whether it is left out", () => {
    const rows = folderPreview(
      ["Northwind Pricing Study/report.pdf", "Northwind Pricing Study/notes.txt", "readme.txt", "Contract ACME-0042/terms.pdf"],
      folder,
    );
    expect(rows).toEqual([
      { path: "Contract ACME-0042/terms.pdf", folder: "Contract ACME-0042", name: "Contract ACME-0042", excluded: false },
      {
        path: "Northwind Pricing Study/notes.txt",
        folder: "Northwind Pricing Study",
        name: "Northwind Pricing Study",
        excluded: true,
      },
      {
        path: "Northwind Pricing Study/report.pdf",
        folder: "Northwind Pricing Study",
        name: "Northwind Pricing Study",
        excluded: false,
      },
      { path: "readme.txt", folder: null, name: null, excluded: false },
    ]);
    expect(folderCounts(rows)).toEqual([
      { folder: "Contract ACME-0042", files: 1 },
      { folder: "Northwind Pricing Study", files: 1 },
    ]);
  });

  it("names the entity by the folder's own name, so folders of one name at the level are one", () => {
    expect(folderName("Picked Root/Northwind Pricing Study")).toBe("Northwind Pricing Study");
    const rows = folderPreview(
      ["Picked Root/Northwind Pricing Study/report.pdf", "Archive/Northwind Pricing Study/old.pdf"],
      { ...folder, level: 2, excludedPaths: [] },
    );
    expect(folderCounts(rows)).toEqual([{ folder: "Northwind Pricing Study", files: 2 }]);
  });

  it("leaves a document out by its file id or by its path", () => {
    expect(folderExcluded(folder, { fileId: "file-x" })).toBe(true);
    expect(folderExcluded(folder, { fileId: "file-y", location: "/Northwind Pricing Study/notes.txt" })).toBe(true);
    expect(folderExcluded(folder, { fileId: "file-y", location: "Northwind Pricing Study/report.pdf" })).toBe(false);
    expect(folderExcluded(folder, { fileId: "file-y" })).toBe(false);
  });
});

describe("a folder entity the pipeline can build", () => {
  const selection: OntologySelection = {
    entityTypes: [{ name: "STUDY", description: "" }, { name: "PRODUCT", description: "" }],
    relationTypes: [{ name: "MADE_BY", description: "" }],
    relations: "guided",
  };

  it("names an entity type of the vocabulary and a relation extraction does not read", () => {
    expect(folderEntityProblem(selection)).toBeNull();
    expect(folderEntityProblem({ ...selection, folderEntity: folder })).toBeNull();
    expect(folderEntityProblem({ ...selection, folderEntity: { ...folder, entityType: "" } })).toMatch(/Choose/);
    expect(folderEntityProblem({ ...selection, folderEntity: { ...folder, entityType: "CONTRACT" } })).toMatch(
      /not one of the entity types/,
    );
    expect(folderEntityProblem({ ...selection, folderEntity: { ...folder, relationType: "made-by" } })).toMatch(
      /reads from text/,
    );
  });
});

describe("the level a folder entity starts at", () => {
  const picked = [
    "Picked Root/README.md",
    "Picked Root/Northwind Pricing Study/report.pdf",
    "Picked Root/Northwind Pricing Study/raw/prices.txt",
    "Picked Root/Contract ACME-0042/terms.pdf",
  ];

  it("skips every folder all files share, to the first folder that tells them apart", () => {
    expect(sharedFolderDepth(picked)).toBe(1);
    expect(suggestedFolderLevel(picked)).toBe(2);
    expect(suggestedFolderLevel(["Drop/Inner/a/x.pdf", "Drop/Inner/b/y.pdf"])).toBe(3);
    // Files from no shared folder, or none at all, start at the top folder.
    expect(suggestedFolderLevel(["Alpha/x.pdf", "Beta/y.pdf"])).toBe(1);
    expect(suggestedFolderLevel([])).toBe(1);
    // Files that all sit in one folder are tied to it, not to a level none reaches.
    expect(suggestedFolderLevel(["Root/Study A/x.pdf", "Root/Study A/y.pdf"])).toBe(2);
    expect(suggestedFolderLevel(["x.pdf"])).toBe(1);
  });

  it("offers each level with a folder the uploads hold at it", () => {
    expect([...levelExamples(picked).entries()]).toEqual([
      [1, "Picked Root"],
      [2, "Contract ACME-0042"],
      [3, "raw"],
    ]);
  });
});

describe("paths and names the folder setting reads", () => {
  it("previews an over-deep path exactly where the upload route stores it", () => {
    const deep = Array.from({ length: 20 }, (_, index) => `level-${index + 1}`).join("/") + "/report.pdf";
    expect(uploadedPath(deep)).toBe(safeUploadPath(deep));
    expect(uploadedPath(deep).split("/")).toHaveLength(16);
    expect(uploadedPath(deep).startsWith("level-6/")).toBe(true);
    expect(uploadedPath(" Picked Root /a\0b/x.pdf")).toBe(safeUploadPath(" Picked Root /a\0b/x.pdf"));
  });

  it("names the folder relation once typing ends, like a relation type", () => {
    expect(folderRelationName("  filed under ")).toBe("FILED_UNDER");
    expect(folderRelationName("   ")).toBe("APPEARS_IN");
    expect(folderRelationName("!!")).toBe("!!");
    const selection: OntologySelection = {
      entityTypes: [{ name: "STUDY", description: "" }],
      relationTypes: [],
      relations: "open",
      folderEntity: { ...folder, relationType: folderRelationName("!!") },
    };
    expect(folderEntityProblem(selection)).toMatch(/capitals, digits and underscores/);
  });
});
