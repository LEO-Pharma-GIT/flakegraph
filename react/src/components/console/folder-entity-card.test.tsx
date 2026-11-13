import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DocumentTable } from "@/components/console/document-table";
import { FolderEntityCard } from "@/components/console/folder-entity-card";
import type { DocumentStatus } from "@/server/documents";
import type { OntologySelection } from "@/server/protocol/schema";

const ontology: OntologySelection = {
  entityTypes: [{ name: "STUDY", description: "" }, { name: "PRODUCT", description: "" }],
  relationTypes: [{ name: "MADE_BY", description: "" }],
  relations: "guided",
};
const folderEntity = {
  entityType: "STUDY",
  relationType: "APPEARS_IN",
  level: 1,
  excludedFileIds: [],
  excludedPaths: ["Northwind Pricing Study/notes.txt"],
};
const noop = () => undefined;

describe("the folder setting on the ingestion form", () => {
  it("is off until switched on", () => {
    const html = renderToStaticMarkup(<FolderEntityCard ontology={ontology} onChange={noop} paths={[]} />);
    expect(html).toContain("Tie documents to their folder");
    expect(html).not.toContain("Folder entity type");
  });

  it("previews the folder each uploaded file is tied to, with a box to leave it out", () => {
    const html = renderToStaticMarkup(
      <FolderEntityCard
        ontology={{ ...ontology, folderEntity }}
        onChange={noop}
        paths={["Northwind Pricing Study/report.pdf", "Northwind Pricing Study/notes.txt", "readme.txt"]}
      />,
    );
    expect(html).toContain("Folder entity type");
    expect(html).toContain("1 folder at this level: Northwind Pricing Study (1)");
    expect(html).toContain("folders of one name at this level");
    expect(html).toContain('aria-label="Tie Northwind Pricing Study/report.pdf to its folder" checked=""');
    expect(html).not.toContain('aria-label="Tie Northwind Pricing Study/notes.txt to its folder" checked=""');
    expect(html).toContain("No folder at this level");
  });

  it("names each file's folder by its own name and says which shared folder the level skips", () => {
    const html = renderToStaticMarkup(
      <FolderEntityCard
        ontology={{ ...ontology, folderEntity: { ...folderEntity, level: 2, excludedPaths: [] } }}
        onChange={noop}
        paths={["Picked Root/Northwind Pricing Study/report.pdf", "Picked Root/Contract ACME-0042/terms.pdf"]}
      />,
    );
    expect(html).toContain("2 folders at this level: Contract ACME-0042 (1), Northwind Pricing Study (1)");
    expect(html).toContain('title="Picked Root/Northwind Pricing Study">Northwind Pricing Study<');
    expect(html).toContain("Every file is in Picked Root, so the folders that tell them apart start at level 2.");
  });

  it("asks for the entity type the folder names before a run can start", () => {
    const html = renderToStaticMarkup(
      <FolderEntityCard ontology={{ ...ontology, folderEntity: { ...folderEntity, entityType: "" } }} onChange={noop} paths={[]} />,
    );
    expect(html).toContain("Choose the entity type a document&#x27;s folder names.");
  });
});

describe("a graph's documents and their folders", () => {
  const documents: DocumentStatus[] = [
    { fileId: "f1", name: "report.pdf", location: "Northwind Pricing Study/report.pdf", phase: "indexed", detail: "Indexed" },
    { fileId: "f2", name: "notes.txt", location: "notes.txt", phase: "indexed", detail: "Indexed" },
  ];

  it("shows each document's folder and a box that unties it in the next version", () => {
    const html = renderToStaticMarkup(
      <DocumentTable documents={documents} folder={{ level: 1, untied: new Set(["f1"]), onToggle: noop }} />,
    );
    expect(html).toContain(">Folder<");
    expect(html).toContain('aria-label="Exclude report.pdf from folder Northwind Pricing Study" checked=""');
    expect(html).toContain("No folder at this level");
    expect(renderToStaticMarkup(<DocumentTable documents={documents} />)).not.toContain(">Folder<");
  });
});
