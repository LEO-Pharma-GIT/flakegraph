import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/** Where the e2e server keeps its state; the seed writes fixtures there. */
const STATE_ROOT = process.env.FLAKEGRAPH_APP_STATE_ROOT ?? path.join(tmpdir(), "flakegraph-e2e");

/**
 * Each journey starts as a session that assumed nobody. The assumed identity
 * is workspace state that outlives a test, and graphs are private to their
 * owner, so a journey that signed in as CAROL would otherwise hand CAROL's
 * view of the catalog to the next one.
 */
test.beforeEach(async ({ request }) => {
  const response = await request.post("/api/trpc/auth.assume", {
    headers: { "content-type": "application/json" },
    data: { json: { userName: "", roles: [], role: "operator" } },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
});

/** Assume a demo principal through the identity dialog. */
async function signInAs(page: Page, name: "ALICE" | "BOB" | "CAROL") {
  await page.getByTestId("identity-chip").click();
  await page.getByRole("button", { name: `Sign in as ${name}` }).click();
  await expect(page.getByTestId("identity-chip")).toContainText(name);
}

async function confirmEnvironmentIfNeeded(page: Page) {
  const confirm = page.getByRole("button", { name: "Confirm environment" });
  await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
  if (await confirm.isVisible()) {
    await confirm.click();
    await expect(confirm).toHaveCount(0);
  }
}

/** Pick where the documents come from; the choices are always on screen. */
async function chooseSource(page: Page, name: string) {
  const tile = page.getByRole("group", { name: "Document source" }).getByRole("button", { name, exact: true });
  await expect(tile).toBeVisible();
  if ((await tile.getAttribute("aria-pressed")) !== "true") {
    await tile.click();
  }
}

async function useSamplePack(page: Page, name: "Martial arts" | "Deep learning papers" = "Martial arts") {
  await chooseSource(page, "Sample pack");
  const tile = page.getByRole("button", { name, exact: true });
  await expect(tile).toBeVisible();
  if ((await tile.getAttribute("aria-pressed")) !== "true") {
    await tile.click();
  }
}

// React Query parks retries while the document is hidden; this flips what
// it reads without needing a second window.
async function setVisibility(page: Page, state: "hidden" | "visible") {
  await page.evaluate((next) => {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => next });
    Object.defineProperty(document, "hidden", { configurable: true, get: () => next === "hidden" });
    document.dispatchEvent(new Event("visibilitychange", { bubbles: true }));
  }, state);
}

/** Open the Explore toolbar's Filters popover and hand back its locator. */
async function openFilters(page: Page) {
  const filters = page.getByTestId("explore-filters");
  if (!(await filters.isVisible())) {
    await page.getByRole("button", { name: "Filters", exact: true }).click();
  }
  await expect(filters).toBeVisible();
  return filters;
}

async function useFolderPath(page: Page, path: string) {
  await chooseSource(page, "Folder path");
  await page.getByLabel("Folder path").fill(path);
}

test.describe("local martial arts graph", () => {
  test("lists the seeded graph and opens the explorer", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await expect(page.getByRole("heading", { name: "Martial arts history" })).toBeVisible();
    await expect(page.getByTestId("status-sentence")).toContainText("Ready to explore");
    await expect(page.getByRole("tab", { name: "Quality" })).toBeVisible();
    // One Export menu carries every download, each item saying what it holds.
    await page.getByRole("button", { name: "Export" }).click();
    await expect(page.getByRole("menuitem", { name: /Subgraph as JSON/ })).toContainText("the whole graph");
    await expect(page.getByRole("menuitem", { name: /Review bundle/ })).toBeVisible();
    await expect(page.getByRole("menuitem", { name: /Consumption/ })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("guide-card")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Show neighborhood" })).toHaveCount(0);
    await page.getByRole("tab", { name: "Quality" }).click();
    await expect(page.getByTestId("gold-card")).toContainText(/required relation/i);
    // The pack's gold is scored by the evaluator, beside the results recorded on it.
    const benchmark = page.getByTestId("benchmark-card");
    await expect(benchmark).toContainText("martial-arts-history-v1");
    // The pack's gold is exhaustive, so precision and F1 are scored too.
    await expect(benchmark.getByTestId("benchmark-entities")).toContainText("Precision1.000");
    await expect(benchmark.getByTestId("benchmark-entities")).toContainText("F11.000");
    await expect(benchmark.getByTestId("benchmark-reference-note")).toHaveCount(0);
    await expect(benchmark.getByTestId("benchmark-gates")).toContainText("Entity recall");
    const baselines = benchmark.getByTestId("benchmark-baselines");
    await expect(baselines.getByRole("row", { name: /This graph/ })).toBeVisible();
    await expect(baselines).toContainText("unsloth/Qwen3.8-27B-NVFP4");
    await page.getByRole("tab", { name: "Explore" }).click();
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    // The tables are a tab of their own, narrowed by what Explore searches.
    await page.getByPlaceholder("Entity name or description").fill("Kano");
    await page.getByRole("tab", { name: "Data" }).click();
    await expect(page.getByRole("tab", { name: "Entities" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "Jigoro Kano" }).first()).toBeVisible();
  });

  test("zooms, fits, and recolors the graph explorer", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Explore" }).click();
    const canvas = page.getByTestId("graph-canvas");
    await expect(canvas).toBeVisible();
    // The mouse hints live on the toolbar's help icon, not a line of their own.
    await expect(page.getByRole("img", { name: /scroll to zoom/i })).toBeVisible();
    const before = await canvas.getAttribute("data-zoom");
    await page.getByRole("button", { name: "Zoom in" }).click();
    await expect.poll(async () => canvas.getAttribute("data-zoom")).not.toBe(before);
    await page.getByRole("button", { name: "Fit graph" }).click();
    await page.getByRole("tab", { name: "Degree" }).click();
    await expect(page.getByRole("tab", { name: "Degree" })).toHaveAttribute("data-state", "active");
    await page.getByRole("button", { name: "Show minimap" }).click();
    await expect(page.getByTestId("graph-minimap")).toBeVisible();
    await expect(page.getByRole("button", { name: "Export GraphML" })).toHaveCount(0);
  });

  test("exports the focused subgraph as JSON and GraphML", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByPlaceholder("Entity name or description").fill("judo");
    await page.getByRole("button", { name: "Export" }).click();
    const [json] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("menuitem", { name: /Subgraph as JSON/ }).click(),
    ]);
    expect(json.suggestedFilename()).toMatch(/-subgraph\.json$/);
    const subgraph = JSON.parse(await readFile(await json.path(), "utf8")) as { nodes: unknown[]; edges: unknown[] };
    expect(subgraph.nodes.length).toBeGreaterThan(0);
    expect(subgraph.nodes.length).toBeLessThan(20);
    await page.getByRole("button", { name: "Export" }).click();
    const [graphml] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("menuitem", { name: /GraphML/ }).click(),
    ]);
    expect(graphml.suggestedFilename()).toMatch(/\.graphml$/);
    const xml = await readFile(await graphml.path(), "utf8");
    expect(xml).toContain("<graphml");
    expect(xml.match(/<node /g)?.length).toBe(subgraph.nodes.length);
    expect(xml.match(/<edge /g)?.length).toBe(subgraph.edges.length);
  });

  test("renames a graph from the workspace", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_rename_target");
    await page.getByRole("button", { name: "Rename", exact: true }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByLabel("Graph name").fill("Martial arts corpus");
    await page.getByRole("button", { name: "Save name" }).click();
    await expect(page.getByRole("heading", { name: "Martial arts corpus" })).toBeVisible();
  });

  test("filters entities, relations, neighborhoods, and consumption", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    const filters = await openFilters(page);
    await filters.getByTestId("facet-entity-types").getByRole("button", { name: "Choose entity types" }).click();
    const panel = page.getByTestId("facet-entity-types-panel");
    await panel.getByRole("checkbox", { name: /^Person/ }).check();
    await panel.getByRole("button", { name: "Done" }).click();
    // The Data tab reads the same focus and says what narrows it.
    await page.getByRole("tab", { name: "Data" }).click();
    await expect(page.getByTestId("graph-data-scope")).toContainText("1 entity type");
    await expect(page.getByRole("cell", { name: "Jigoro Kano" }).first()).toBeVisible();
    await expect(page.getByRole("cell", { name: "Judo", exact: true })).toHaveCount(0);
    await page.getByTestId("graph-data-scope").getByRole("button", { name: "Clear filters" }).click();
    await expect(page.getByTestId("graph-data-scope")).toHaveCount(0);
    await expect(page.getByRole("cell", { name: "Judo", exact: true }).first()).toBeVisible();
    await page.getByRole("tab", { name: "Relations" }).click();
    await expect(page.getByRole("cell", { name: "DEVELOPED_BY" }).first()).toBeVisible();
    await page.getByRole("tab", { name: "Neighborhoods" }).click();
    await expect(page.getByRole("cell", { name: "PERSON" }).first()).toBeVisible();
    await page.getByRole("tab", { name: "Run details" }).click();
    await expect(page.getByTestId("consumption")).toContainText(/usd/i);
  });

  test("narrows the canvas by entity and relation type from Filters", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    const scope = page.getByTestId("explore-scope");
    await expect(scope).toContainText("119 of 119 entities · 148 relations");
    await expect(page.getByTestId("filters-count")).toHaveCount(0);
    const filters = await openFilters(page);
    const summary = page.getByTestId("filters-summary");
    await expect(summary).toHaveText("Filters");
    await filters.getByTestId("facet-entity-types").getByRole("button", { name: "Choose entity types" }).click();
    const entityTypes = page.getByTestId("facet-entity-types-panel");
    // Ranked by how many entities carry the type: 28 techniques before 17 locations, 16 martial arts and 11 people.
    await expect(entityTypes.getByRole("checkbox").first()).toHaveAccessibleName(/^Technique/);
    await expect(entityTypes.getByRole("checkbox").nth(2)).toHaveAccessibleName(/^Martial Art/);
    await expect(entityTypes.getByRole("checkbox").nth(4)).toHaveAccessibleName(/^Person/);
    await entityTypes.getByRole("checkbox", { name: /^Person/ }).check();
    await entityTypes.getByRole("button", { name: "Done" }).click();
    await expect(scope).toContainText("11 of 11 entities · 2 relations");
    await filters.getByTestId("facet-relation-types").getByRole("button", { name: "Choose relation types" }).click();
    const relationTypes = page.getByTestId("facet-relation-types-panel");
    await relationTypes.getByLabel("Find relation types").fill("studied");
    await expect(relationTypes).toContainText("1 of 22 relation types");
    await relationTypes.getByRole("checkbox", { name: /^Studied Under/ }).check();
    await relationTypes.getByRole("button", { name: "Done" }).click();
    await expect(scope).toContainText("11 of 11 entities · 1 relation");
    await expect(
      filters.getByTestId("facet-relation-types").getByRole("button", { name: "Remove Studied Under" }),
    ).toBeVisible();
    // The popover's header says what narrows the graph and can undo it; the
    // toolbar button carries the count once the popover is closed.
    await expect(summary).toContainText("1 entity type · 1 relation type");
    await page.keyboard.press("Escape");
    await expect(filters).toBeHidden();
    await expect(page.getByTestId("filters-count")).toHaveText("2");
    await openFilters(page);
    await summary.getByRole("button", { name: "Clear filters" }).click();
    await expect(summary).toHaveText("Filters");
    await expect(scope).toContainText("119 of 119 entities · 148 relations");
    await expect(page.getByTestId("filters-count")).toHaveCount(0);
    // Claimed relations show by default; switched off, only reported ones stay.
    await openFilters(page);
    await expect(filters.getByTestId("claimed-count")).toHaveText("(1)");
    await filters.getByRole("switch", { name: "Include claimed relations" }).click();
    await expect(scope).toContainText("119 of 119 entities · 147 relations");
    await expect(summary).toContainText("reported only");
    await summary.getByRole("button", { name: "Clear filters" }).click();
    await expect(scope).toContainText("119 of 119 entities · 148 relations");
  });

  test("searches the sidebar catalog across datasets", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Search graphs").fill("Deep learning");
    await expect(page.getByRole("button", { name: /Deep learning papers Ready to/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Martial arts history Ready to/ })).toHaveCount(0);
  });

  test("explains a catalog search miss instead of looking empty", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Search graphs").fill("xyzzy-no-such-graph");
    await expect(page.getByTestId("catalog-empty")).toContainText("xyzzy-no-such-graph");
  });

  test("deletes a graph after confirming, with everything the console kept for it", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_delete_me");
    await expect(page.getByRole("heading", { name: "Disposable graph" })).toBeVisible();
    // The confirmation says what goes, and Cancel keeps everything.
    await page.getByRole("button", { name: "Delete Disposable graph" }).click();
    const dialog = page.getByTestId("delete-graph-dialog");
    await expect(dialog).toContainText("Delete “Disposable graph”?");
    await expect(dialog).toContainText("cannot be undone");
    await expect(dialog.getByTestId("delete-graph-scope")).toContainText("1 run");
    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByRole("button", { name: "Delete Disposable graph" })).toBeVisible();
    expect(existsSync(path.join(STATE_ROOT, "runs", "run_delete_me"))).toBe(true);

    await page.getByRole("button", { name: "Delete Disposable graph" }).click();
    await dialog.getByRole("button", { name: "Delete graph" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByText("Deleted “Disposable graph”")).toBeVisible();
    await expect(page.getByRole("button", { name: "Delete Disposable graph" })).toHaveCount(0);
    // Gone from disk, not only from the list.
    expect(existsSync(path.join(STATE_ROOT, "runs", "run_delete_me"))).toBe(false);
    expect(existsSync(path.join(STATE_ROOT, "graphs", "graph_delete_me"))).toBe(false);
    await page.goto("/?runtime=local&page=run&run=run_delete_me");
    await expect(page.getByRole("heading", { name: "Run not found" })).toBeVisible();
  });
});

test.describe("deep learning papers graph", () => {
  test("opens a second dataset with different counts", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_deep_learning");
    await expect(page.getByRole("heading", { name: "Deep learning papers" })).toBeVisible();
    await page.getByPlaceholder("Entity name or description").fill("ImageNet");
    await page.getByRole("tab", { name: "Data" }).click();
    await expect(page.getByRole("tab", { name: "Entities" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "ImageNet" }).first()).toBeVisible();
    const martial = await page.request.get("/api/health");
    expect(martial.ok()).toBeTruthy();
  });
});

test.describe("ingestion", () => {
  test("lists a local corpus and runs preflight", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await expect(page.getByRole("group", { name: "Document source" }).getByRole("button", { name: "Upload" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveCount(0);
    await useSamplePack(page);
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("source-count")).toContainText("selectable object", { timeout: 20_000 });
    await page.getByRole("button", { name: "Preview configuration" }).click();
    await expect(page.getByText("Effective configuration")).toBeVisible();
    await page.getByRole("button", { name: "Run preflight" }).click();
    await expect(page.getByRole("alert").filter({ hasText: "Preflight passed" })).toBeVisible();
  });

  test("submits a fake worker and reaches a completed graph", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await page.getByLabel("Display name").fill("Smoke graph");
    await confirmEnvironmentIfNeeded(page);
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Smoke graph" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("main").getByText("succeeded", { exact: true }).first()).toBeVisible({
      timeout: 30_000,
    });
  });

  test("uploads a markdown file and lists it as a source object", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await page.getByLabel("Upload documents", { exact: true }).setInputFiles({
      name: "note.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("Judo was developed by Jigoro Kano.\n"),
    });
    await expect(page.getByTestId("source-count")).toContainText("selectable object", { timeout: 20_000 });
  });

  test("uploads a folder with its subfolders and leaves hidden files out", async ({ page }) => {
    const root = await mkdtemp(path.join(tmpdir(), "flakegraph-folder-"));
    const corpus = path.join(root, "corpus");
    await mkdir(path.join(corpus, "judo", "kodokan"), { recursive: true });
    await mkdir(path.join(corpus, ".git"), { recursive: true });
    await writeFile(path.join(corpus, "README.md"), "Martial arts corpus.\n");
    await writeFile(path.join(corpus, "judo", "README.md"), "Judo was developed by Jigoro Kano.\n");
    await writeFile(path.join(corpus, "judo", "kodokan", "history.md"), "The Kodokan opened in 1882.\n");
    await writeFile(path.join(corpus, ".DS_Store"), "litter");
    await writeFile(path.join(corpus, ".git", "config"), "[core]\n");

    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Upload a folder", { exact: true }).setInputFiles(corpus);
    await expect(page.getByText(/Uploaded 3 files from 3 folders · skipped 2 hidden or system files/)).toBeVisible({
      timeout: 20_000,
    });
    // Two README.md files in different folders stay two documents.
    await expect(page.getByTestId("source-count")).toContainText("3 selectable objects", { timeout: 20_000 });
    await expect(page.getByTestId("file-dropzone")).toContainText("3 files ready");
  });

  test("ties uploaded documents to their folder and previews the folder of each file", async ({ page }) => {
    const root = await mkdtemp(path.join(tmpdir(), "flakegraph-folders-"));
    const corpus = path.join(root, "corpus");
    await mkdir(path.join(corpus, "Northwind Pricing Study", "raw"), { recursive: true });
    await mkdir(path.join(corpus, "Contract ACME-0042"), { recursive: true });
    await writeFile(path.join(corpus, "README.md"), "Studies and contracts.\n");
    await writeFile(path.join(corpus, "Northwind Pricing Study", "report.md"), "Widget Pro prices rose.\n");
    await writeFile(path.join(corpus, "Northwind Pricing Study", "raw", "prices.md"), "Widget Pro sold for less.\n");
    await writeFile(path.join(corpus, "Contract ACME-0042", "terms.md"), "Acme Corp supplies Gadget Max.\n");

    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Upload a folder", { exact: true }).setInputFiles(corpus);
    await expect(page.getByTestId("file-dropzone")).toContainText("4 files ready", { timeout: 20_000 });

    const card = page.getByTestId("folder-entity");
    await card.getByRole("switch", { name: "Tie documents to their folder" }).click();
    await expect(card).toContainText("Choose the entity type a document's folder names.");
    await card.getByLabel("Folder entity type").click();
    await page.getByRole("option", { name: "WORK", exact: true }).click();
    // Every file is in the picked folder, so the level starts below it, at
    // the folders that tell the files apart, and each is named by its own name.
    const level = card.getByLabel("Folder level");
    await expect(level).toHaveText("2 · e.g. Contract ACME-0042");
    await expect(card).toContainText("Every file is in corpus, so the folders that tell them apart start at level 2.");
    const summary = card.getByTestId("folder-preview-summary");
    await expect(summary).toHaveText("2 folders at this level: Contract ACME-0042 (1), Northwind Pricing Study (2)");
    // The relation field keeps what is typed, and names it once typing ends.
    const relation = card.getByLabel("Folder relation");
    await relation.fill("filed under ");
    await expect(relation).toHaveValue("filed under ");
    await relation.blur();
    await expect(relation).toHaveValue("FILED_UNDER");
    await expect(card.getByLabel("Tie corpus/README.md to its folder")).toBeDisabled();
    await card.getByLabel("Tie corpus/Northwind Pricing Study/raw/prices.md to its folder").uncheck();
    await expect(summary).toHaveText("2 folders at this level: Contract ACME-0042 (1), Northwind Pricing Study (1)");

    // Each level is offered with a folder the uploads hold at it.
    await level.click();
    await expect(page.getByRole("option", { name: "1 · e.g. corpus" })).toBeVisible();
    await expect(page.getByRole("option", { name: "4 · no file is this deep" })).toBeVisible();
    await page.getByRole("option", { name: "3 · e.g. raw" }).click();
    await expect(summary).toHaveText("0 folders at this level");
    // A file added at the upload root shares no folder with the rest; the
    // level someone chose stays.
    await page.getByLabel("Upload documents", { exact: true }).setInputFiles({
      name: "extra.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("An extra note.\n"),
    });
    await expect(page.getByTestId("file-dropzone")).toContainText("5 files ready", { timeout: 20_000 });
    await expect(level).toHaveText("3 · e.g. raw");
  });

  test("shows an upload's progress while it runs and lets it be cancelled", async ({ page }) => {
    // A slow link: the server takes its time answering each batch.
    await page.route("**/api/uploads", async (route) => {
      await new Promise((settle) => setTimeout(settle, 4_000));
      await route.continue().catch(() => undefined);
    });
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Upload documents", { exact: true }).setInputFiles([
      { name: "a.md", mimeType: "text/markdown", buffer: Buffer.alloc(2048, "a") },
      { name: "b.md", mimeType: "text/markdown", buffer: Buffer.alloc(2048, "b") },
    ]);
    await expect(page.getByTestId("file-dropzone")).toContainText(/Uploading .* of .* · 0 of 2 files in the folder/);
    await expect(page.getByRole("progressbar", { name: "Upload progress" })).toBeVisible();
    await page.getByRole("button", { name: "Cancel upload" }).click();
    await expect(page.getByText("Upload cancelled.")).toBeVisible();
    await expect(page.getByTestId("file-dropzone")).toContainText("Drop files or folders here");
  });

  test("a drop without folder entries still uploads its files", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    const dropzone = page.getByTestId("file-dropzone");
    await expect(dropzone).toBeVisible();
    const data = await page.evaluateHandle(() => {
      const transfer = new DataTransfer();
      transfer.items.add(new File(["Karate came from Okinawa.\n"], "karate.md", { type: "text/markdown" }));
      transfer.items.add(new File(["Aikido was founded by Morihei Ueshiba.\n"], "aikido.md", { type: "text/markdown" }));
      return transfer;
    });
    await dropzone.dispatchEvent("drop", { dataTransfer: data });
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
  });

  test("shows Azure and S3 source fields from local capabilities", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await chooseSource(page, "Azure Blob");
    await expect(page.getByLabel("Account URL")).toBeVisible();
    await expect(page.getByText("Blobs are listed once an account URL and container are named.")).toBeVisible();
    await chooseSource(page, "S3-compatible bucket");
    await expect(page.getByLabel("Bucket")).toBeVisible();
    await expect(page.getByText("Objects are listed once a bucket is named.")).toBeVisible();
  });

  test("lists a bucket before Start and holds Start to what it lists", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await chooseSource(page, "S3-compatible bucket");
    const start = page.getByRole("button", { name: "Start", exact: true });
    await expect(start).toBeDisabled();
    await page.getByLabel("Bucket").fill("test-corpora");
    await page.getByLabel("Prefix").fill("martial_arts/");
    // The fake CLI answers `sources list` with two supported objects.
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
    await expect(page.getByTestId("source-count")).toContainText("41 KB");
    await expect(page.getByTestId("credit-envelope")).toBeVisible({ timeout: 20_000 });
    await expect(start).toBeEnabled();
    // A refused listing disables Start, including in a hidden tab. Provider
    // diagnostics stay server-side; the browser receives a safe action hint.
    await setVisibility(page, "hidden");
    await page.getByLabel("Bucket").fill("missing");
    await expect(page.getByText(/Listing the s3 source failed/)).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText(/NoSuchBucket/)).not.toBeVisible();
    await expect(start).toBeDisabled();
    await setVisibility(page, "visible");
    await expect(page.getByText(/Listing failed/)).toBeVisible();
    await expect(start).toBeDisabled();
    // And clearing the bucket leaves nothing to list.
    await page.getByLabel("Bucket").fill("   ");
    await expect(page.getByText("Objects are listed once a bucket is named.")).toBeVisible();
    await expect(start).toBeDisabled();
  });
});

test.describe("unavailable and failed runs", () => {
  test("explains missing artifacts instead of opening the explorer", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_missing_artifacts");
    await expect(page.getByText("files are not available on this host")).toBeVisible();
  });

  test("shows a failed run error", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_failed_llm");
    await expect(page.getByTestId("status-sentence")).toContainText("LLM endpoint timed out");
    // The catalog row carries that error on one line; it must truncate in
    // its column rather than widen the list into a sideways scroll.
    const overflow = await page.evaluate(() => {
      const viewport = document.querySelector("[data-testid='catalog-list']");
      return viewport ? viewport.scrollWidth - viewport.clientWidth : -1;
    });
    expect(overflow).toBe(0);
    await expect(page.getByTestId("guide-card")).toContainText("This run did not finish");
    await expect(page.getByTestId("guide-card")).toContainText("Clone the config");
    await expect(page.getByRole("button", { name: "Delete this graph" })).toBeVisible();
    await expect(page.getByRole("alert").filter({ hasText: "LLM endpoint timed out" })).toHaveCount(0);
    await page.getByRole("button", { name: "New graph from this config" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByLabel("Display name")).toHaveValue("LLM timeout");
    await expect(page.getByRole("button", { name: "Sample pack", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByLabel("LLM provider")).toContainText("Ollama");
    await expect(page.getByLabel("OCR provider")).toContainText("Built-in document text only");
  });
});

test.describe("active progress", () => {
  test("shows live OCR progress for an in-flight run", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_active_ocr");
    await expect(page.getByRole("heading", { name: "Active OCR run" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Progress" })).toBeVisible();
    await expect(page.getByText(/ocr/i).first()).toBeVisible();
  });
});

test.describe("kubernetes fleet", () => {
  test("shows stub nodes", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=fleet");
    await expect(page.getByRole("heading", { name: "Compute fleet" })).toBeVisible();
    // The dashboards are one click away when the deployment names them.
    await expect(page.getByRole("link", { name: "Open Grafana" })).toHaveAttribute("href", "https://grafana.example.test");
    await expect(page.getByRole("heading", { name: "gpu-a" })).toBeVisible();
    await expect(page.getByText("Fleet martial arts")).toBeVisible();
    // The console manages the fleet it runs in; there is no cluster to pick.
    await expect(page.getByRole("button", { name: "Clusters" })).toHaveCount(0);
  });

  test("keeps a large fleet readable", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=fleet");
    // Four numbers describe the fleet however many pods it runs.
    const summary = page.getByTestId("fleet-summary");
    await expect(summary).toContainText("Nodes ready");
    await expect(summary).toContainText("3 / 4");
    await expect(summary).toContainText("Workers running");
    await expect(summary).toContainText("37");
    await expect(summary).toContainText("Pending / failed");
    await expect(summary).toContainText("2 / 1");
    // Four nodes still fit as cards; a busy node lists a page of leases and counts the rest.
    await expect(page.getByTestId("fleet-node-grid")).toBeVisible();
    await expect(page.getByRole("heading", { name: "gpu-d" })).toBeVisible();
    const leases = page.getByTestId("node-assignments-gpu-b");
    await expect(leases).toContainText("23 leased tasks");
    await expect(leases.getByRole("listitem")).toHaveCount(20);
    await expect(leases).toContainText("3 more");
    // The list opens on active pods with trouble first, a page at a time.
    const table = page.getByTestId("fleet-workloads");
    const count = page.getByTestId("fleet-workloads-count");
    await expect(count).toContainText("58 active of 62 workloads");
    await expect(table.getByRole("row")).toHaveCount(51);
    const rows = table.getByRole("row");
    await expect(rows.nth(1)).toContainText("worker-extract-oom-7");
    await expect(rows.nth(1)).toContainText("Failed · 3 restarts");
    await expect(rows.nth(2)).toContainText("worker-extract-24");
    await expect(rows.nth(2)).toContainText("Waiting for node capacity (3/4 ready)");
    await expect(rows.nth(4)).toContainText("ocr-shim-1");
    await expect(rows.nth(4)).toContainText("Not ready · 5 restarts");
    await expect(table).toContainText("Showing 50 of 58");
    await table.getByRole("button", { name: /Show 8 more/ }).click();
    await expect(table.getByRole("row")).toHaveCount(59);
    await expect(table.getByRole("button", { name: /Show \d+ more/ })).toHaveCount(0);
    // Finished Jobs wait behind the phase filter and read as completed, not as a fault.
    await table.getByLabel("Phase filter").click();
    await page.getByRole("option", { name: "Succeeded · 4" }).click();
    await expect(count).toContainText("4 of 62 workloads");
    await expect(table.getByRole("row")).toHaveCount(5);
    await expect(table).toContainText("bench-export-p8k2z");
    await expect(table.getByRole("cell", { name: "Completed" })).toHaveCount(4);
    await expect(table).not.toContainText("not ready");
    await table.getByLabel("Phase filter").click();
    await page.getByRole("option", { name: "Active · 58" }).click();
    // Component and search narrow the same list.
    await table.getByLabel("Component filter").click();
    await page.getByRole("option", { name: "Monitoring · 3" }).click();
    await expect(count).toContainText("3 of 62 workloads");
    await expect(table.getByRole("row")).toHaveCount(4);
    await table.getByLabel("Component filter").click();
    await page.getByRole("option", { name: /All components/ }).click();
    await table.getByLabel("Search workloads").fill("vllm");
    await expect(count).toContainText("3 of 62 workloads");
    await expect(table).toContainText("vllm-a");
    await expect(table).not.toContainText("vllm28-probe");
    await table.getByLabel("Search workloads").fill("");
    // A node card narrows the list to that node; the chip clears it.
    await page.getByRole("button", { name: "Show workloads on gpu-b" }).click();
    await expect(page.getByRole("button", { name: "Show workloads on gpu-b" })).toHaveAttribute("aria-pressed", "true");
    await expect(count).toContainText("of 62 workloads");
    await expect(table.getByRole("row").filter({ hasNotText: "gpu-b" })).toHaveCount(1);
    await expect(table.getByRole("row").nth(1)).toContainText("worker-extract-oom-7");
    await page.getByRole("button", { name: "Clear node filter gpu-b" }).click();
    await expect(count).toContainText("58 active of 62 workloads");
  });

  test("submits a stub fleet job", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    await confirmEnvironmentIfNeeded(page);
    await useSamplePack(page);
    await page.getByLabel("Display name").fill("Fleet smoke");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Fleet smoke" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
  });

  test("clones a bucket source back into compose with every field", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    await confirmEnvironmentIfNeeded(page);
    await chooseSource(page, "S3-compatible bucket");
    await page.getByLabel("Bucket", { exact: true }).fill("test-corpora");
    await page.getByLabel("Prefix", { exact: true }).fill("martial_arts/");
    await page.getByLabel("Endpoint", { exact: true }).fill("http://minio.local:9000");
    await page.getByLabel("Region", { exact: true }).fill("us-east-1");
    await page.getByLabel("Display name").fill("Bucket clone");
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Bucket clone" })).toBeVisible({ timeout: 30_000 });
    // A run page hands its configuration back to compose with the bucket as
    // it was named, not just its kind.
    await page.getByTestId("guide-card").getByRole("button", { name: "Cancel job" }).click();
    await expect(page.getByRole("main").getByText("cancelled", { exact: true }).first()).toBeVisible();
    await page.getByTestId("guide-card").getByRole("button", { name: "New graph from this config" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByLabel("Bucket", { exact: true })).toHaveValue("test-corpora");
    await expect(page.getByLabel("Prefix", { exact: true })).toHaveValue("martial_arts/");
    await expect(page.getByLabel("Endpoint", { exact: true })).toHaveValue("http://minio.local:9000");
    await expect(page.getByLabel("Region", { exact: true })).toHaveValue("us-east-1");
    await expect(page.getByLabel("Display name")).toHaveValue("Bucket clone");
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
  });

  test("builds a new version of a finished fleet graph from its Edit tab", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await expect(page.getByRole("heading", { name: "Fleet judo" })).toBeVisible();
    await page.getByRole("tab", { name: "Edit" }).click();
    const editor = page.getByTestId("graph-editor");
    await expect(editor.getByText("karate-history.md")).toBeVisible();
    const confirm = page.getByRole("button", { name: "Confirm environment" });
    if (await confirm.isVisible()) {
      await confirm.click();
    }
    // Nothing removed and nothing added: there is no version to build yet.
    await expect(page.getByTestId("revision-summary")).toContainText("Keeps 2 documents");
    await expect(page.getByRole("button", { name: "Build new version" })).toBeDisabled();
    // The whole set can be left out at once and kept again; then one is enough on its own.
    await editor.getByRole("button", { name: /Remove all/ }).click();
    await expect(page.getByTestId("removal-summary")).toContainText("2 of 2 documents will be left out");
    await editor.getByRole("button", { name: "Keep all" }).click();
    await expect(page.getByTestId("removal-summary")).toHaveCount(0);
    await page.getByLabel("Remove karate-history.md").check();
    await expect(page.getByTestId("removal-summary")).toContainText("1 of 2 documents will be left out");
    await expect(page.getByTestId("revision-summary")).toContainText("removes 1");
    await expect(page.getByRole("button", { name: "Build new version" })).toBeEnabled();
    // Adding a file is counted too, and the form is the compose form: no name, no ontology.
    await page.getByLabel("Upload documents", { exact: true }).setInputFiles({
      name: "aikido.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("Aikido was developed by Morihei Ueshiba.\n"),
    });
    await expect(page.getByTestId("revision-summary")).toContainText("adds 1", { timeout: 20_000 });
    await expect(page.getByLabel("Display name")).toHaveCount(0);
    await page.getByRole("button", { name: "Build new version" }).click();
    await expect(page.getByText(/Building a new version of Fleet judo/)).toBeVisible();
    // The new run keeps the graph's name and identity and is queued on the fleet.
    await expect(page.getByRole("heading", { name: "Fleet judo" })).toBeVisible({ timeout: 30_000 });
    await expect(page).not.toHaveURL(/run=run_k8s_done/);
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
    await expect(page.getByRole("main")).toContainText("graph_k8s_done");
  });

  test("removing a document alone is enough to build a new version, and versions are listed", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await expect(page.getByRole("heading", { name: "Fleet judo" })).toBeVisible();
    await page.getByRole("tab", { name: "Versions" }).click();
    // The fleet keeps the versions: two runs published this graph, the later
    // one is the head, and the one on screen says so. The workspace's own
    // version labels do not appear on the fleet.
    const versions = page.getByTestId("graph-versions");
    await expect(versions).toContainText("v2");
    await expect(versions).toContainText("head");
    await expect(versions.getByRole("row").filter({ hasText: "run_k8s_done" }).filter({ hasText: "viewing" })).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Publish new version" })).toHaveCount(0);
    await expect(page.getByText("Watch new files")).toHaveCount(0);
    // Opening the head from the list lands on that run; opening the older
    // one again shows it is behind the head before it is edited.
    await versions.getByRole("row").filter({ hasText: "run_k8s_done_v2" }).getByRole("button", { name: "Open" }).click();
    await expect(page).toHaveURL(/run=run_k8s_done_v2/);
    await page.getByRole("tab", { name: "Edit" }).click();
    await expect(page.getByTestId("editing-behind-head")).toHaveCount(0);
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await page.getByRole("tab", { name: "Edit" }).click();
    await expect(page.getByTestId("editing-behind-head")).toContainText("version 1 of 2, not the head");
    await page.getByLabel("Remove judo-history.md").check();
    const confirm = page.getByRole("button", { name: "Confirm environment" });
    if (await confirm.isVisible()) {
      await confirm.click();
    }
    await expect(page.getByTestId("revision-summary")).toContainText("Keeps 1 document · removes 1");
    // Removing the last one too, with nothing added, is refused: a version
    // needs at least one document.
    await page.getByLabel("Remove karate-history.md").check();
    await expect(page.getByTestId("revision-summary")).toContainText("a version needs at least one document");
    await expect(page.getByRole("button", { name: "Build new version" })).toBeDisabled();
    await page.getByLabel("Remove karate-history.md").uncheck();
    await expect(page.getByRole("button", { name: "Build new version" })).toBeEnabled();
    await page.getByRole("button", { name: "Build new version" }).click();
    await expect(page.getByText(/Building a new version of Fleet judo/)).toBeVisible();
    await expect(page).not.toHaveURL(/run=run_k8s_done/);
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible({ timeout: 30_000 });
  });

  test("pages a graph with many versions, head first, the viewed one next", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_kata_03");
    await expect(page.getByRole("heading", { name: "Revised kata" })).toBeVisible();
    await page.getByRole("tab", { name: "Versions" }).click();
    const versions = page.getByTestId("graph-versions");
    await expect(page.getByTestId("graph-versions-count")).toHaveText("14 versions");
    // A header row and a page of ten: the head, then the version on screen,
    // then the rest newest first.
    const rows = versions.getByRole("row");
    await expect(rows).toHaveCount(11);
    await expect(rows.nth(1)).toContainText("v14");
    await expect(rows.nth(1)).toContainText("head");
    await expect(rows.nth(2)).toContainText("v3");
    await expect(rows.nth(2)).toContainText("viewing");
    await expect(rows.nth(2).getByRole("button", { name: "Open" })).toHaveCount(0);
    await expect(rows.nth(3)).toContainText("v13");
    await expect(versions).toContainText("Showing 10 of 14 versions");
    await versions.getByRole("button", { name: "Show 4 more" }).click();
    await expect(rows).toHaveCount(15);
    await expect(versions.getByTestId("show-more")).toHaveCount(0);
    await expect(rows.last()).toContainText("v1");
    await rows.nth(1).getByRole("button", { name: "Open" }).click();
    await expect(page).toHaveURL(/run=run_k8s_kata_14/);
  });

  test("cancels an in-flight fleet run and retries a failed one", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_martial");
    await expect(page.getByRole("heading", { name: "Fleet martial arts" })).toBeVisible();
    await page.getByRole("button", { name: "Recover workers" }).click();
    await expect(page.getByText(/Reconciled infrastructure/i)).toBeVisible();
    await page.getByTestId("guide-card").getByRole("button", { name: "Cancel job" }).click();
    await expect(page.getByRole("main").getByText("cancelled", { exact: true }).first()).toBeVisible();
    // A cancelled run keeps what it finished and can be picked up again.
    await expect(page.getByTestId("guide-card")).toContainText("Resume this run where it stopped");
    await page.getByTestId("guide-card").getByRole("button", { name: "Resume run" }).click();
    await expect(page.getByText("Run resumed")).toBeVisible();
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_failed");
    await expect(page.getByTestId("status-sentence")).toContainText("Worker lease expired");
    await page.getByTestId("guide-card").getByRole("button", { name: "Retry failed documents" }).click();
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
  });

  test("switches from local to kubernetes without wiping compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Display name").fill("Keep me");
    await page.getByLabel("Runtime").click();
    await page.getByRole("option", { name: "Kubernetes" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByLabel("Display name")).toHaveValue("Keep me");
    await expect(page.getByTestId("identity-chip")).toContainText("Kubernetes");
    const ack = await page.evaluate(() => sessionStorage.getItem("flakegraph.runtime-ack"));
    expect(ack).not.toBe("kubernetes");
    await page.getByLabel("Runtime").click();
    await page.getByRole("option", { name: "Snowflake" }).click();
    await expect(page.getByLabel("Display name")).toHaveValue("Keep me");
    await expect(page.getByRole("button", { name: "Confirm environment" })).toBeVisible();
  });
});

test.describe("catalog length", () => {
  test("a long catalog extends the page instead of scrolling inside the sidebar", async ({ page }) => {
    // Short enough that the seeded catalog cannot fit: the page must grow and
    // scroll as a whole, and the list itself must never become a scroll box.
    await page.setViewportSize({ width: 1280, height: 520 });
    await page.goto("/?runtime=local&page=new");
    const list = page.getByTestId("catalog-list");
    await expect(list).toBeVisible();
    const measured = await page.evaluate(() => {
      const element = document.querySelector("[data-testid='catalog-list']") as HTMLElement;
      const root = document.scrollingElement as HTMLElement;
      return {
        listScrollable: element.scrollHeight - element.clientHeight,
        pageScrollable: root.scrollHeight - window.innerHeight,
      };
    });
    expect(measured.listScrollable).toBe(0);
    expect(measured.pageScrollable).toBeGreaterThan(0);

    // Scrolling the page reaches the last graph in the catalog.
    const rows = list.locator("li, [role='option'], button").filter({ hasText: /documents|Ready|Queued|Running|Failed/ });
    await rows.last().scrollIntoViewIfNeeded();
    await expect(rows.last()).toBeInViewport();
    expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(0);
  });

  test("on a small screen the navigation drawer still scrolls within itself", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 640 });
    await page.goto("/?runtime=local&page=new");
    await page.getByRole("button", { name: "Open navigation" }).click();
    const list = page.getByTestId("catalog-list");
    await expect(list).toBeVisible();
    const overflowY = await list.evaluate((element) => getComputedStyle(element).overflowY);
    expect(overflowY).toBe("auto");
  });
});

test.describe("identity and catalog scope", () => {
  test("opens the identity dialog and explains delete", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await expect(page.getByRole("dialog", { name: "Session identity" })).toBeVisible();
    await expect(page.getByRole("dialog")).toContainText("removes a graph for good");
    await expect(page.getByRole("dialog")).toContainText("cannot be undone");
    await expect(page.getByRole("dialog")).toContainText("private to the person who built it");
    await expect(page.getByTestId("identity-chip")).toContainText("not signed in");
  });

  test("a laptop session that signed in as nobody sees every graph, but signs in to share", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("button", { name: "Mine", exact: true })).toHaveAttribute("aria-pressed", "true");
    const catalog = page.getByTestId("catalog-list");
    await expect(catalog.getByText("Martial arts history")).toBeVisible();
    await expect(catalog.getByText("Bob field notes")).toBeVisible();
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByTestId("sharing-sign-in")).toBeVisible();
    await expect(page.getByTestId("sharing-add")).toHaveCount(0);
  });

  test("Mine lists your own graphs and Shared the ones shared with you", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await signInAs(page, "ALICE");
    const catalog = page.getByTestId("catalog-list");
    await expect(page.getByRole("button", { name: "Mine", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(catalog.getByText("Martial arts history")).toBeVisible();
    await expect(catalog.getByText("Bob field notes")).toHaveCount(0);
    await page.getByRole("button", { name: "Shared", exact: true }).click();
    await expect(catalog.getByText("Martial arts history")).toHaveCount(0);
    const row = catalog.getByRole("button", { name: /Bob field notes/ });
    await expect(row).toBeVisible();
    await expect(row.getByTestId("catalog-sharing")).toHaveText("Shared by BOB · Read");
    await expect(page.getByRole("button", { name: "All", exact: true })).toHaveCount(0);
    // Opening a graph shows the scope it belongs to, so its row is in view.
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await expect(page.getByRole("button", { name: "Mine", exact: true })).toHaveAttribute("aria-pressed", "true");
    await page.goto("/?runtime=local&page=run&run=run_bob_notes");
    await expect(page.getByRole("button", { name: "Shared", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(catalog.getByRole("button", { name: /Bob field notes/ })).toBeVisible();
  });
});

test.describe("graph sharing", () => {
  test.describe.configure({ mode: "serial" });

  test("keeps a graph private until its owner shares it, then opens it at the level given", async ({ page }) => {
    // Someone else's graph is not listed and does not open.
    await page.goto("/?runtime=local&page=new");
    await signInAs(page, "BOB");
    await expect(page.getByTestId("catalog-list").getByText("Team handbook")).toHaveCount(0);
    await page.getByRole("button", { name: "Shared", exact: true }).click();
    await expect(page.getByTestId("catalog-list").getByText("Team handbook")).toHaveCount(0);
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await expect(page.getByRole("heading", { name: "You don't have access to this graph" })).toBeVisible();
    await expect(page.getByTestId("no-access")).toContainText("Ask its owner to share it with you");

    // The owner shares it with BOB to read.
    await signInAs(page, "ALICE");
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await expect(page.getByRole("heading", { name: "Team handbook" })).toBeVisible();
    await expect(page.getByTestId("access-badge")).toHaveText("Private");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByTestId("sharing-private")).toBeVisible();
    await expect(page.getByTestId("sharing-levels")).toContainText("Read");
    await expect(page.getByTestId("sharing-levels")).toContainText("Write");
    await page.getByLabel("Person to share with").fill("bob");
    await expect(page.getByLabel("Access level")).toHaveText("Read");
    await page.getByRole("button", { name: "Share", exact: true }).click();
    const people = page.getByTestId("sharing-people");
    await expect(people.locator('[data-principal="BOB"]')).toContainText("Added by ALICE");
    await expect(page.getByLabel("Access for BOB")).toHaveText("Read");
    await expect(page.getByTestId("access-badge").first()).toHaveText("Shared with 1 person");

    // BOB sees it under Shared and can read, not change it.
    await signInAs(page, "BOB");
    await page.getByRole("button", { name: "Shared", exact: true }).click();
    await expect(
      page.getByTestId("catalog-list").getByRole("button", { name: /Team handbook/ }).getByTestId("catalog-sharing"),
    ).toHaveText("Shared by ALICE · Read");
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await expect(page.getByRole("heading", { name: "Team handbook" })).toBeVisible();
    await expect(page.getByTestId("access-badge")).toHaveText("Shared with you · Read");
    await expect(page.getByRole("button", { name: "Rename" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Delete graph" })).toHaveCount(0);
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByTestId("sharing-mine")).toContainText("You have Read access, shared by ALICE");
    await expect(page.getByTestId("sharing-add")).toHaveCount(0);
    await expect(page.getByLabel("Access for BOB")).toHaveCount(0);

    // Raised to Write, BOB may rename it, but still not delete or share it.
    await signInAs(page, "ALICE");
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await page.getByLabel("Access for BOB").click();
    await page.getByRole("option", { name: "Write" }).click();
    await expect(page.getByLabel("Access for BOB")).toHaveText("Write");
    await signInAs(page, "BOB");
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await expect(page.getByTestId("access-badge")).toHaveText("Shared with you · Write");
    await expect(page.getByRole("button", { name: "Rename" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Delete graph" })).toHaveCount(0);

    // BOB leaves; the graph is gone from his list and closed to him again.
    await page.getByRole("tab", { name: "Sharing" }).click();
    await page.getByTestId("sharing-mine").getByRole("button", { name: "Leave" }).click();
    await page.getByRole("dialog").getByRole("button", { name: "Leave graph" }).click();
    await page.getByRole("button", { name: "Shared", exact: true }).click();
    await expect(page.getByTestId("catalog-list").getByText("Team handbook")).toHaveCount(0);
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await expect(page.getByTestId("no-access")).toBeVisible();
  });

  test("lets the owner remove someone, and refuses addresses that are not people", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_share_demo");
    await signInAs(page, "ALICE");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await page.getByLabel("Person to share with").fill("not a person!");
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await expect(page.getByTestId("sharing-add").getByRole("alert")).toBeVisible();
    await page.getByLabel("Person to share with").fill("CAROL");
    await page.getByLabel("Access level").click();
    await page.getByRole("option", { name: "Write" }).click();
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await expect(page.getByLabel("Access for CAROL")).toHaveText("Write");
    // Removing someone is at once, with Undo that gives the same level back.
    await page.getByRole("button", { name: "Remove CAROL" }).click();
    await expect(page.getByTestId("sharing-private")).toBeVisible();
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.getByLabel("Access for CAROL")).toHaveText("Write");
    await page.getByRole("button", { name: "Remove CAROL" }).click();
    await expect(page.getByTestId("sharing-private")).toBeVisible();
  });

  test("shows a read-only review queue to a Read grantee", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_bob_notes");
    await signInAs(page, "ALICE");
    await expect(page.getByRole("heading", { name: "Bob field notes" })).toBeVisible();
    await page.getByRole("tab", { name: "Review" }).click();
    await expect(page.getByText("You have read access")).toBeVisible();
    await expect(page.getByRole("button", { name: "Sample 1% high-confidence" })).toHaveCount(0);
  });
});

test.describe("snowflake sharing", () => {
  test("offers the same Sharing tab on a stored graph", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_martial");
    await signInAs(page, "ALICE");
    await expect(page.getByRole("heading", { name: "Shared martial arts" })).toBeVisible();
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByTestId("sharing-add")).toBeVisible();
  });

  test("closes a graph BOB owns to anyone he has not shared it with", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=new");
    await signInAs(page, "ALICE");
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_private");
    await expect(page.getByRole("heading", { name: "You don't have access to this graph" })).toBeVisible();
  });

  test("deletes a stored graph", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_delete");
    await expect(page.getByRole("heading", { name: "Disposable snowflake graph" })).toBeVisible();
    await page.getByRole("button", { name: "Delete graph" }).click();
    await expect(page.getByRole("dialog")).toContainText("cannot be undone");
    await page.getByRole("dialog").getByRole("button", { name: "Delete graph" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
  });

  test("submits a stub snowflake job from an uploaded document", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=new");
    await confirmEnvironmentIfNeeded(page);
    await expect(page.getByLabel("Stage", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Sample pack", exact: true })).toHaveCount(0);
    await expect(page.getByLabel("LLM model")).toHaveValue("mistral-large2");
    await page.getByLabel("Display name").fill("Snowflake smoke");
    await chooseSource(page, "Upload");
    await page.getByLabel("Upload documents", { exact: true }).setInputFiles({
      name: "note.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("Judo was developed by Jigoro Kano.\n"),
    });
    await expect(page.getByTestId("source-count")).toContainText("1 selectable");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Snowflake smoke" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Recover workers" })).toHaveCount(0);
  });

  test("disables Recover workers on a queued Snowflake job", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_queued");
    await expect(page.getByRole("heading", { name: "Queued Snowflake job" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Recover workers" })).toHaveCount(0);
    await expect(page.getByTestId("guide-card")).toContainText("has not started");
  });

  test("shares a graph whatever its quality says", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_thin");
    await signInAs(page, "ALICE");
    await expect(page.getByRole("heading", { name: "Empty share candidate" })).toBeVisible();
    // Quality reports the problem; it never closes Sharing.
    await expect(page.getByTestId("guide-card")).toContainText("entity count is 0");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByText(/not ready to share/i)).toHaveCount(0);
    await expect(page.getByTestId("sharing-add")).toBeVisible();
  });
});

test.describe("remaining report journeys", () => {
  test("serves inspect HTML for the martial arts graph", async ({ request }) => {
    const response = await request.get("/api/inspect?run=run_martial_arts");
    expect(response.ok()).toBeTruthy();
    expect(await response.text()).toContain("CLI inspect artifact");
  });

  test("asks a local question and opens the Judo perspective", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Ask" }).click();
    // The graph's own community reports supply the starting questions, the
    // first of them already in the box; the planner picks the mode.
    await expect(page.getByRole("button", { name: "Auto", exact: true })).toHaveAttribute("aria-pressed", "true");
    const starters = page.getByTestId("ask-starters");
    await expect(starters.getByRole("button")).toHaveCount(4);
    const first = await starters.getByRole("button").first().textContent();
    await expect(page.getByLabel("Ask the graph")).toHaveValue(first ?? "");
    await starters.getByRole("button").nth(1).click();
    await expect(page.getByLabel("Ask the graph")).not.toHaveValue(first ?? "");
    await page.getByRole("button", { name: "Local", exact: true }).click();
    await page.getByLabel("Ask the graph").fill("Who developed judo?");
    await page.getByRole("button", { name: "Ask", exact: true }).click();
    await expect(page.getByTestId("ask-answer")).toBeVisible({ timeout: 120_000 });
    await page.getByRole("tab", { name: "Explore" }).click();
    await page.getByRole("button", { name: "Perspectives" }).click();
    await page.getByRole("menuitem", { name: /Judo/ }).first().click();
    await expect(page.getByPlaceholder("Entity name or description")).toHaveValue("judo");
    // The perspective's neighborhood counts as a filter.
    await expect(page.getByTestId("filters-count")).toHaveText("1");
    await openFilters(page);
    await expect(page.getByTestId("filters-summary")).toContainText("1 neighborhood");
  });

  test("saves a named perspective and reopens it", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByPlaceholder("Entity name or description").fill("judo");
    // A perspective is a saved filter set, so it is saved from the Filters popover.
    await openFilters(page);
    const save = page.getByRole("button", { name: "Save as perspective" });
    // Nothing to save until the perspective is named.
    await expect(save).toBeDisabled();
    await page.getByLabel("Perspective name").fill("Judo lens");
    await save.click();
    await expect(page.getByText("Perspective saved as draft", { exact: false })).toBeVisible();
    await expect(page.getByLabel("Perspective name")).toHaveValue("");
    await page.keyboard.press("Escape");
    await page.getByPlaceholder("Entity name or description").fill("");
    await page.getByRole("button", { name: "Perspectives" }).click();
    await page.getByRole("menuitem", { name: "Judo lens · draft" }).click();
    await expect(page.getByPlaceholder("Entity name or description")).toHaveValue("judo");
  });

  test("shows a credit envelope on compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await expect(page.getByTestId("credit-envelope")).toBeVisible();
    await expect(page.getByRole("button", { name: "Scan for PII" })).toBeVisible();
    await expect(page.getByLabel("Describe the graph you want")).toBeVisible();
  });

  test("selects a sample pack and required providers on compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    const sourceGroup = page.getByRole("group", { name: "Document source" });
    await expect(sourceGroup.getByRole("button", { name: "Upload" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    // Every source is one click away; nothing is folded behind a disclosure.
    for (const name of ["Upload", "Sample pack", "Folder path", "Azure Blob", "S3-compatible bucket"]) {
      await expect(sourceGroup.getByRole("button", { name, exact: true })).toBeVisible();
    }
    await expect(page.locator("summary")).toHaveCount(0);
    await expect(page.getByText("Run like last time")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Fast / cheap" })).toHaveCount(0);
    await expect(page.getByTestId("compose-providers")).toBeVisible();
    // A fresh form defaults to what a laptop can run: no OCR engine, a
    // small Ollama model, and the sentence-transformers embedder.
    await expect(page.getByLabel("OCR provider")).toContainText("Built-in document text only");
    await expect(page.getByLabel("LLM provider")).toContainText("Ollama");
    await expect(page.getByLabel("LLM model")).toHaveValue("qwen3:4b");
    await expect(page.getByLabel("Embeddings model")).toHaveValue("Qwen/Qwen3-Embedding-0.6B");
    await expect(page.getByLabel("Embeddings dimension")).toHaveValue("1024");
    await useSamplePack(page);
    await expect(sourceGroup.getByRole("button", { name: "Sample pack", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("file-dropzone")).toHaveCount(0);
    await expect(page.getByLabel("OCR provider")).toContainText("Built-in document text only");
    await expect(page.getByLabel("LLM model")).toHaveValue("qwen3:4b");
    await page.getByLabel("LLM provider").click();
    await page.getByRole("option", { name: "vLLM" }).click();
    await expect(page.getByLabel("LLM model")).toHaveValue("Qwen/Qwen3-8B");
    // Only packs whose files are on this host are offered: the papers have
    // to be downloaded first, so they are not.
    await expect(page.getByRole("button", { name: "Deep learning papers", exact: true })).toHaveCount(0);
    await expect(page.getByLabel("Display name")).toHaveValue("Martial arts history");
    await expect(page.getByLabel("LLM model")).toHaveValue("Qwen/Qwen3-8B");
    await chooseSource(page, "Upload");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveCount(0);
  });

  test("chooses what to extract: types, relations, reset, and the run carries them", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await confirmEnvironmentIfNeeded(page);
    const editor = page.getByTestId("ontology-editor");
    const entityTypes = editor.getByTestId("entity-types");
    // A sample pack brings the vocabulary its gold is written in.
    await expect(editor).toContainText("These are the types from the Martial arts pack's ontology.");
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toBeVisible();
    await expect(editor.getByTestId("relation-types").getByRole("button", { name: "FOUNDED_BY", exact: true })).toBeVisible();
    // Any other source has the default profile, as editable chips.
    await useFolderPath(page, path.join(STATE_ROOT, "pii-pack"));
    await expect(editor).toContainText("These are the types from the default profile.");
    await expect(entityTypes.getByRole("button", { name: "PERSON", exact: true })).toBeVisible();
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toHaveCount(0);
    // Add one (normalised to a type name), remove one.
    await entityTypes.getByLabel("Add entity type").fill("martial technique");
    await entityTypes.getByLabel("Add entity type").press("Enter");
    await expect(entityTypes.getByRole("button", { name: "MARTIAL_TECHNIQUE", exact: true })).toBeVisible();
    await entityTypes.getByRole("button", { name: "Remove DATE" }).click();
    await expect(entityTypes.getByRole("button", { name: "DATE", exact: true })).toHaveCount(0);
    // Describe a type.
    await entityTypes.getByRole("button", { name: "MARTIAL_TECHNIQUE", exact: true }).click();
    await entityTypes.getByLabel("Description of MARTIAL_TECHNIQUE").fill("A named throw, strike, or hold.");
    await entityTypes.getByRole("button", { name: "Save" }).click();
    await expect(editor).toContainText("Changed from the default profile.");
    // Relations can be detected automatically, in which case no list is asked for.
    await editor.getByRole("radio", { name: /Detected automatically/ }).click();
    await expect(editor.getByTestId("relation-types")).toHaveCount(0);
    // A fixed list with nothing in it cannot be built with.
    await editor.getByRole("radio", { name: /Only this list/ }).click();
    const relationTypes = editor.getByTestId("relation-types");
    for (const name of ["RELATED_TO", "PART_OF", "LOCATED_IN", "CREATED_BY", "INFLUENCED_BY", "OCCURRED_AT"]) {
      await relationTypes.getByRole("button", { name: `Remove ${name}` }).click();
    }
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeDisabled();
    await expect(editor).toContainText("Add at least one relation type");
    await relationTypes.getByLabel("Add relation type").fill("TRAINED_UNDER");
    await relationTypes.getByLabel("Add relation type").press("Enter");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    // The run's configuration carries exactly this vocabulary, inline.
    await page.getByRole("button", { name: "Preview configuration" }).click();
    const yaml = page.getByLabel("YAML configuration text");
    await expect(yaml).toContainText("MARTIAL_TECHNIQUE");
    await expect(yaml).toContainText("A named throw, strike, or hold.");
    await expect(yaml).toContainText("mode: closed");
    await expect(yaml).toContainText("TRAINED_UNDER");
    await expect(yaml).not.toContainText("name: DATE");
    // Reset returns to the defaults, and the vocabulary follows the source again.
    await editor.getByRole("button", { name: "Reset to the default profile" }).click();
    await expect(entityTypes.getByRole("button", { name: "DATE", exact: true })).toBeVisible();
    await expect(editor).toContainText("These are the types from the default profile.");
    await useSamplePack(page);
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toBeVisible();
    // The pack's run keeps what its profile says beyond the names.
    await page.getByRole("button", { name: "Preview configuration" }).click();
    await expect(page.getByLabel("YAML configuration text")).toContainText("name: martial-arts-history");
    await expect(page.getByLabel("YAML configuration text")).toContainText("evidence_cues:");
  });

  test("a revision keeps the vocabulary of the version it revises", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await page.getByRole("tab", { name: "Edit" }).click();
    const editor = page.getByTestId("ontology-editor");
    await expect(editor).toContainText("built with the vocabulary of the version it revises");
    await expect(editor.getByTestId("entity-types")).toContainText("SCHOOL");
    await expect(editor.getByTestId("relation-types")).toContainText("FOUNDED_BY");
    await expect(editor.getByLabel("Add entity type")).toHaveCount(0);
    await expect(editor.getByRole("button", { name: "Remove SCHOOL" })).toHaveCount(0);
  });

  test("offers to publish a fleet graph to Snowflake only where the fleet has an account", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await page.getByRole("button", { name: "Export" }).click();
    // The stub fleet holds no Snowflake account: the item is there, closed, and says why.
    const item = page.getByRole("menuitem", { name: /Snowflake/ });
    await expect(item).toContainText("names no Snowflake account");
    await expect(item).toHaveAttribute("data-disabled", "");
    await page.keyboard.press("Escape");
    // A local graph has no fleet to publish through, so the section is absent.
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("button", { name: "Export" }).click();
    await expect(page.getByRole("menuitem", { name: /Snowflake/ })).toHaveCount(0);
  });

  test("offers Snowflake as a destination only where an account can write", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    await confirmEnvironmentIfNeeded(page);
    // The stub fleet holds no Snowflake account: the option is there, closed, and says why.
    await expect(page.getByTestId("snowflake-unavailable")).toContainText("no Snowflake account");
    await page.getByLabel("Destination").click();
    await expect(page.getByRole("option", { name: "Snowflake" })).toHaveAttribute("data-disabled", "");
    await page.keyboard.press("Escape");
    // Locally the account is named on the form, and Start waits for the target.
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await confirmEnvironmentIfNeeded(page);
    await page.getByLabel("Destination").click();
    await page.getByRole("option", { name: "Snowflake" }).click();
    await expect(page.getByTestId("snowflake-destination")).toBeVisible();
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeDisabled();
    await expect(page.getByText("Name the Snowflake database, schema and bulk stage before Start.")).toBeVisible();
    await page.getByLabel("Account").fill("xy123");
    await page.getByLabel("User").fill("ALICE");
    await page.getByLabel("Database").fill("FG");
    await page.getByLabel("Schema").fill("PUBLIC");
    await page.getByLabel("Bulk Stage").fill("@stage");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Preview configuration" }).click();
    await expect(page.getByLabel("YAML configuration text")).toContainText("provider: snowflake_bulk");
  });

  test("logo returns home from a graph workspace", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await expect(page.getByRole("heading", { name: "Martial arts history" })).toBeVisible();
    await page.getByRole("button", { name: "FlakeGraph home" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
  });

  test("opens the sidebar on a narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("button", { name: "Open navigation" })).toBeVisible();
    await page.getByRole("button", { name: "Open navigation" }).click();
    await expect(page.getByRole("button", { name: "New graph" })).toBeVisible();
  });

  test("skips a poison document on an active run", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_active_ocr");
    // The document needing a decision is listed on the Progress card itself.
    const attention = page.getByTestId("documents-needing-attention");
    await expect(attention).toContainText("One document needs a decision");
    await expect(attention).toContainText("doc_poison_scan");
    await attention.getByRole("button", { name: "Skip file" }).click();
    // Skipped, it needs nothing more: the list goes, the summary counts it.
    await expect(page.getByTestId("documents-needing-attention")).toHaveCount(0);
    await expect(page.getByTestId("phase-summary")).toContainText("Skipped");
    await page.getByText(/^All \d+ documents$/).click();
    await expect(page.getByTestId("document-table")).toContainText("doc_poison_scan");
    await expect(page.getByTestId("document-table")).toContainText("Quarantined");
  });

  test("analyst chrome hides New graph after CAROL signs in", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign in as CAROL" }).click();
    await expect(page.getByRole("button", { name: "New graph" })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Perspectives" })).toBeVisible();
    await page.getByRole("button", { name: "Open Judo" }).click();
    await expect(page.getByRole("heading", { name: "Martial arts history" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Rename", exact: true })).toHaveCount(0);
    await page.getByRole("tab", { name: "Versions" }).click();
    await expect(page.getByText("Watch new files")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Publish new version" })).toHaveCount(0);
    await expect(page.getByLabel("Runtime")).toHaveCount(0);
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page.getByRole("button", { name: "New graph" })).toBeVisible();
  });

  test("opens inspect HTML with a gold compare section", async ({ request }) => {
    const response = await request.get("/api/inspect?run=run_martial_arts");
    const body = await response.text();
    expect(body).toContain("Gold compare");
  });

  test("reviews an uncertain triple", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Review" }).click();
    await expect(page.getByText(/never rewrite the stored graph/i)).toBeVisible();
    await expect(page.getByText("rel_judo_developed_by")).toBeVisible();
    await page.getByRole("button", { name: "Keep", exact: true }).click();
    await expect(page.getByTestId("review-decision")).toContainText("Recorded keep");
    // A relabel names the relation the triple should carry, from the graph's own vocabulary.
    await page.getByRole("button", { name: "Relabel…" }).click();
    const relabel = page.getByTestId("relabel-editor");
    await expect(page.getByRole("button", { name: "Record relabel" })).toBeDisabled();
    await relabel.getByLabel("New relation type").fill("founded by");
    await expect(relabel.getByLabel("New relation type")).toHaveValue("FOUNDED_BY");
    await page.getByRole("button", { name: "Record relabel" }).click();
    await expect(page.getByTestId("review-decision")).toContainText("Recorded relabel to FOUNDED_BY");
    // A merge says which end is the duplicate and of whom.
    await page.getByRole("button", { name: "Merge…" }).click();
    const merge = page.getByTestId("merge-editor");
    await merge.getByLabel("Which entity is the duplicate").selectOption("source");
    await merge.getByLabel("Merge into").fill("Kodokan judo");
    await page.getByRole("button", { name: "Record merge" }).click();
    await expect(page.getByTestId("review-decision")).toContainText("the source is a duplicate of Kodokan judo");
  });

  test("reports a few extraction gaps down to the page and the text that was lost", async ({ page }) => {
    await page.goto("/?page=run&run=run_deep_learning");
    // The header says so before anyone opens Quality, and takes them there.
    const link = page.getByTestId("gaps-link");
    await expect(link).toContainText("3 extraction gaps in 2 documents");
    await link.click();
    const card = page.getByTestId("extraction-gaps");
    await expect(card.getByTestId("extraction-gaps-badge")).toHaveText(/3 windows · 2 of 49 documents/);
    await expect(card.getByTestId("extraction-gaps-stats")).toContainText("Entity windows2");
    await expect(card.getByTestId("extraction-gaps-reasons")).toContainText("Ungrounded Quote · 3");
    await expect(card.getByTestId("record-table")).toContainText("2 documents");
    // One document's windows, each with its pages, reasons and the text.
    await card.getByRole("button", { name: /^Show windows:/ }).first().click();
    const gaps = card.getByTestId("document-gaps");
    await expect(gaps).toContainText("2 windows · pages 1, 4–5 · 4 records rejected");
    await expect(gaps).toContainText("An introduction to our capabilities");
    await expect(gaps).toContainText("Pages 4–5 · 3 records rejected · Domain Or Range Violation 2 · Self Loop 1");
    // A kind filter narrows the table without losing the totals.
    await card.getByLabel("Gap kind").click();
    await page.getByRole("option", { name: /Relation windows · 1/ }).click();
    await expect(card.getByTestId("record-table")).toContainText("1 document (of 2 total)");
  });

  test("names a document that could not be read, and why, without failing the graph", async ({ page }) => {
    await page.goto("/?page=run&run=run_deep_learning");
    // The header says so and leads to the report.
    const link = page.getByTestId("failed-documents-link");
    await expect(link).toContainText("1 document could not be read");
    await link.click();
    const card = page.getByTestId("failed-documents");
    await expect(card.getByTestId("failed-documents-badge")).toHaveText("1 of 50 documents");
    await expect(card.getByTestId("failed-documents-errors")).toContainText("RuntimeError · 1");
    const table = card.getByTestId("record-table");
    await expect(table).toContainText("broken.pdf");
    await expect(table).toContainText("RuntimeError: mineru_api returned HTTP 400");
    // The whole reason is a click away when it is longer than the cell.
    await table.getByRole("button", { name: "More" }).click();
    await expect(table).toContainText('{"detail":"Unsupported file type: txt"}');
    // The graph built from the rest is there to explore.
    await expect(page.getByText("1 document could not be read").first()).toBeVisible();
  });

  test("stays readable with a thousand extraction gaps across hundreds of documents", async ({ page }) => {
    await page.goto("/?page=run&run=run_gappy_corpus");
    await page.getByRole("tab", { name: "Quality" }).click();
    const card = page.getByTestId("extraction-gaps");
    await expect(card.getByTestId("extraction-gaps-badge")).toHaveText(/1,178 windows · 420 of 600 documents/);
    // Fifty documents a page, the biggest loss first, and the rest on demand.
    const table = card.getByTestId("record-table");
    await expect(table).toContainText("Showing 50 of 420 documents");
    await expect(table.getByRole("row").nth(1)).toContainText("record-0006.pdf");
    await table.getByRole("button", { name: "Show 50 more" }).click();
    await expect(table).toContainText("Showing 100 of 420 documents");
    // Search finds one document among hundreds.
    await table.getByLabel("Find in documents").fill("record-0324");
    await expect(table).toContainText("1 document (of 420 total)");
    await table.getByLabel("Find in documents").fill("");
    // A document with 160 windows lists a page at a time and says what it left out.
    await card.getByRole("button", { name: /^Show windows: record-0006\.pdf/ }).click();
    const gaps = card.getByTestId("document-gaps");
    await expect(gaps).toContainText("160 windows");
    await expect(gaps).toContainText("Showing 10 of 100 windows");
    await gaps.getByRole("button", { name: "Show 10 more" }).click();
    await expect(gaps).toContainText("Showing 20 of 100 windows");
    await expect(gaps).toContainText("60 more windows are counted above but not listed here");
    // Run details marks every document that carries a gap and can filter to them.
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      card.getByRole("button", { name: "Download as JSON" }).click(),
    ]);
    expect(download.suggestedFilename()).toBe("graph_gappy_corpus-extraction-gaps.json");
  });

  test("marks the documents that carry gaps on Run details", async ({ page }) => {
    await page.goto("/?page=run&run=run_martial_thin");
    await page.getByRole("tab", { name: "Run details" }).click();
    const table = page.getByTestId("document-table");
    await expect(table.getByTestId("document-gaps").first()).toHaveText("2 gaps");
    await table.getByLabel("Phase filter").click();
    await page.getByRole("option", { name: /With extraction gaps · 2/ }).click();
    await expect(table.getByTestId("document-table-count")).toHaveText("2 of 20 documents");
  });

  test("uploads a gold file, is held to it, and can let it go", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await page.getByRole("tab", { name: "Quality" }).click();
    const card = page.getByTestId("gold-card");
    await expect(card).toContainText("None yet");
    await expect(card).toContainText("No gold file for this graph");
    // The format is shown, and a template comes from the graph itself.
    await card.getByText("The format", { exact: true }).click();
    await expect(card.getByTestId("gold-format")).toContainText('"relation_type"');
    const [template] = await Promise.all([
      page.waitForEvent("download"),
      card.getByRole("button", { name: "Download a template from this graph" }).click(),
    ]);
    expect(template.suggestedFilename()).toMatch(/-gold-template\.json$/);
    const draft = JSON.parse(await readFile(await template.path(), "utf8")) as { entities: unknown[]; relations: unknown[] };
    expect(draft.entities.length).toBeGreaterThan(0);
    // A file that does not follow the format is refused with the field named.
    await card.getByLabel("Gold file").setInputFiles({
      name: "bad.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify({ entities: [{ id: "x" }] })),
    });
    await expect(page.getByText('A gold file needs a string "name".')).toBeVisible();
    // A good one is kept and the graph is held to it: one required relation missing.
    await card.getByLabel("Gold file").setInputFiles({
      name: "gold.json",
      mimeType: "application/json",
      buffer: Buffer.from(
        JSON.stringify({
          name: "Judo QA",
          description: "What the judo documents must yield.",
          entities: [
            { id: "judo", name: "Judo", type: "MARTIAL_ART" },
            { id: "kano", name: "Jigoro Kano", type: "PERSON" },
            { id: "nobody", name: "Nobody Here", type: "PERSON" },
          ],
          relations: [
            { id: "r1", source: "judo", target: "kano", relation_type: "DEVELOPED_BY", required: true },
            { id: "r2", source: "judo", target: "nobody", relation_type: "TAUGHT_BY", required: true },
          ],
        }),
      ),
    });
    await expect(card).toContainText("Uploaded");
    await expect(card.getByTestId("gold-result")).toContainText("Judo QA");
    await expect(card.getByTestId("gold-result")).toContainText("1/2 required relations found");
    await expect(card.getByTestId("missing-required")).toContainText("Nobody Here");
    // The fleet graph is scored by the evaluator too.
    await expect(page.getByTestId("benchmark-relations")).toContainText("Recall0.500");
    await expect(page.getByTestId("benchmark-miss-reasons")).toContainText("Both ends found, no edge · 1");
    await card.getByRole("button", { name: "Remove" }).click();
    await expect(card).toContainText("None yet");
    await expect(page.getByTestId("benchmark-card")).toHaveCount(0);
  });

  test("pages the workspace's own versions, production first", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Versions" }).click();
    const versions = page.getByTestId("workspace-versions");
    const items = versions.getByRole("listitem");
    await expect(items).toHaveCount(10);
    await expect(items.first()).toContainText("v12");
    await expect(items.first()).toContainText("production");
    await expect(items.nth(1)).toContainText("v11");
    await expect(page.getByText("Showing 10 of 12 versions")).toBeVisible();
    await page.getByRole("button", { name: "Show 2 more" }).click();
    await expect(items).toHaveCount(12);
    await expect(items.last()).toHaveText(/^v1 ·/);
  });

  test("pages the missing gold relations and the run's event log", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_thin");
    await expect(page.getByRole("heading", { name: "Thin martial arts extract" })).toBeVisible();
    await page.getByRole("tab", { name: "Quality" }).click();
    const missing = page.getByTestId("missing-required");
    await expect(missing).toContainText(/\d+ required relations are missing\./);
    await expect(missing.getByRole("listitem")).toHaveCount(12);
    await expect(missing).toContainText(/Showing 12 of \d+ missing relations/);
    await missing.getByRole("button", { name: "Show 12 more" }).click();
    await expect(missing.getByRole("listitem")).toHaveCount(24);
    // The events card is a tail of the log and says so.
    await page.getByRole("tab", { name: "Run details" }).click();
    await expect(page.getByTestId("recent-events-count")).toHaveText("Last 30 of 40 events");
    const details = page.getByTestId("recent-events");
    await expect(details.getByRole("listitem")).toHaveCount(30);
    await details.getByRole("button", { name: "Show 10 more" }).click();
    await expect(details.getByRole("listitem")).toHaveCount(40);
    await expect(page.getByTestId("recent-events-count")).toHaveText("40 events");
  });

  test("ingests new files from a watch", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Versions" }).click();
    await expect(page.getByText("data/martial_arts/files")).toBeVisible();
    await page.getByRole("button", { name: "Ingest new files" }).click();
    await expect(page.getByText(/incremental/i).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Process on compose" })).toBeVisible();
  });

  test("opens the command palette with Cmd+K", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("button", { name: "Jump" })).toBeVisible();
    await page.getByRole("button", { name: "Jump" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Jump" })).toBeVisible();
  });

  test("Jump entity search does not stick after picking another graph", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("button", { name: "Jump" }).click();
    await page.getByLabel("Command palette").fill("Judo");
    await page.getByRole("option", { name: /Entity Judo/ }).click();
    await expect(page.getByLabel("Search entities")).toHaveValue("Judo");
    await page.getByRole("button", { name: /Deep learning papers Ready to/ }).click();
    await expect(page.getByRole("heading", { name: "Deep learning papers" })).toBeVisible();
    await expect(page.getByLabel("Search entities")).toHaveValue("");
  });

  test("staff can build an identity incident note", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Staff" }).click();
    await page.getByRole("button", { name: "Incident" }).click();
    await expect(page.getByRole("heading", { name: "Staff incident" })).toBeVisible();
    await page.getByRole("button", { name: "Build incident note" }).click();
    await expect(page.getByText("Identity mismatch incident")).toBeVisible();
  });

  test("creates an SDK key that does not use SSO", async ({ page }) => {
    await page.goto("/?runtime=local&page=keys");
    await expect(page.getByText(/Machine credentials/i)).toBeVisible();
    // A key acts as whoever minted it, so an unidentified session cannot mint one.
    await expect(page.getByRole("button", { name: "Create a key" })).toBeDisabled();
    await expect(page.getByTestId("sdk-key-sign-in")).toContainText("Sign in first");
    await signInAs(page, "ALICE");
    await page.keyboard.press("Escape");
    await expect(page.getByRole("button", { name: "Create a key" })).toBeEnabled();
    await expect(page.getByRole("heading", { name: "Where the API docs are" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "How to use a key" })).toBeVisible();
    await expect(page.getByRole("link", { name: "tRPC HTTP" })).toHaveAttribute("href", "https://trpc.io/docs/rpc");
    await expect(page.getByTestId("sdk-docs")).toContainText("/api/docs");
    await expect(page.getByTestId("sdk-procedure-catalog")).toContainText("graphs.ask");
    await expect(page.getByRole("tab", { name: "List graphs" })).toHaveAttribute("data-state", "active");
    await expect(page.getByRole("tab", { name: "Python" })).toHaveCount(0);
    await expect(page.getByTestId("sdk-example-language")).toBeVisible();
    await expect(page.getByTestId("sdk-key-example")).toContainText("Authorization: Bearer");
    await expect(page.getByTestId("sdk-key-example")).toContainText("/api/trpc/runs.list");
    await expect(page.getByTestId("sdk-key-example")).not.toContainText("CONTROL_PLANE");
    await expect(page.getByTestId("sdk-key-example")).toContainText("FLAKEGRAPH_API_KEY");
    await page.getByRole("button", { name: "TypeScript", exact: true }).click();
    await expect(page.getByTestId("sdk-key-example")).toContainText("fetch(");
    await page.getByRole("button", { name: "Python", exact: true }).click();
    await expect(page.getByTestId("sdk-key-example")).toContainText("urllib.request");
    await page.getByRole("tab", { name: "Ask a graph" }).click();
    await expect(page.getByTestId("sdk-ask-example")).toContainText("graphs.ask");
    await expect(page.getByTestId("sdk-ask-example")).toContainText("urllib.request");
    await page.getByRole("button", { name: "curl", exact: true }).click();
    await expect(page.getByTestId("sdk-ask-example")).toContainText("/api/trpc/graphs.ask");
    const docs = await page.request.get("/api/docs");
    expect(docs.ok()).toBeTruthy();
    const catalog = await docs.json();
    expect(catalog.endpoint).toContain("/api/trpc");
    expect(catalog.procedures.some((item: { name: string }) => item.name === "runs.list")).toBeTruthy();
    // The seeded deployment already holds more keys than one page: the list
    // says so, pages, and narrows by name.
    const keyList = page.getByTestId("sdk-key-list");
    await expect(page.getByTestId("sdk-key-count")).toHaveText("30 keys");
    await expect(keyList.getByRole("listitem")).toHaveCount(25);
    await expect(keyList).toContainText("Showing 25 of 30 keys");
    await keyList.getByRole("button", { name: "Show 5 more" }).click();
    await expect(keyList.getByRole("listitem")).toHaveCount(30);
    await expect(keyList.getByTestId("show-more")).toHaveCount(0);
    await keyList.getByLabel("Search keys").fill("deploy-bot");
    await expect(page.getByTestId("sdk-key-count")).toHaveText("10 of 30 keys");
    await expect(keyList.getByRole("listitem")).toHaveCount(10);
    await keyList.getByLabel("Search keys").fill("");
    await page.getByRole("button", { name: "Create a key" }).click();
    await expect(page.getByText(/Copy now:/)).toBeVisible();
    const secretLine = await page.getByText(/Copy now:/).innerText();
    const secret = secretLine.replace(/^.*Copy now:\s*/, "").trim();
    expect(secret).toMatch(/^fg_/);
    // The key just made is the newest, so it heads the list.
    await expect(page.getByTestId("sdk-key-count")).toHaveText("31 keys");
    await expect(keyList.getByRole("listitem").first()).toContainText("ci-eval");
    const listed = await page.request.get(
      `/api/trpc/runs.list?input=${encodeURIComponent(JSON.stringify({ json: { limit: 5 } }))}`,
      { headers: { Authorization: `Bearer ${secret}` } },
    );
    expect(listed.ok()).toBeTruthy();
    const denied = await page.request.get(
      `/api/trpc/runs.list?input=${encodeURIComponent(JSON.stringify({ json: { limit: 5 } }))}`,
      { headers: { Authorization: "Bearer fg_revoked_or_wrong" } },
    );
    expect(denied.status()).toBe(401);
    await keyList.getByLabel("Search keys").fill("ci-eval");
    await expect(page.getByTestId("sdk-key-count")).toHaveText("1 of 31 keys");
    await page.getByRole("button", { name: "Revoke ci-eval" }).click();
    await page.getByRole("button", { name: "Confirm revoke ci-eval" }).click();
    await expect(keyList).toContainText("No key matches this search.");
    await expect(page.getByTestId("sdk-key-count")).toHaveText("0 of 30 keys");
    const afterRevoke = await page.request.get(
      `/api/trpc/runs.list?input=${encodeURIComponent(JSON.stringify({ json: { limit: 5 } }))}`,
      { headers: { Authorization: `Bearer ${secret}` } },
    );
    expect(afterRevoke.status()).toBe(401);
  });

  test("deletes selected graphs in bulk after confirming", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    const bar = page.getByTestId("catalog-selection");
    await page.getByLabel("Select Bulk delete me").check();
    await expect(bar).toContainText("1 of");
    // Delete asks first, naming what goes, and Cancel keeps everything.
    await page.getByRole("button", { name: "Delete selected" }).click();
    const dialog = page.getByTestId("delete-graph-dialog");
    await expect(dialog).toContainText("Delete “Bulk delete me”?");
    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByLabel("Select Bulk delete me")).toBeChecked();
    await page.getByRole("button", { name: "Delete selected" }).click();
    await dialog.getByRole("button", { name: "Delete graph" }).click();
    await expect(page.getByLabel("Select Bulk delete me")).toHaveCount(0);
    await expect(bar).toContainText("Select all");
    expect(existsSync(path.join(STATE_ROOT, "runs", "run_bulk_delete"))).toBe(false);
  });

  test("keeps a 300-document run readable while it runs", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_large_corpus");
    // The Progress card sums the corpus up and lists only what needs a decision.
    const summary = page.getByTestId("phase-summary");
    await expect(summary).toContainText("6");
    await expect(summary).toContainText("OCR failed");
    await expect(summary).toContainText("Indexed");
    const attention = page.getByTestId("documents-needing-attention");
    // paper-043 and paper-086 are here only because the whole events file is
    // read: the snapshot's tail of 200 events does not reach them.
    await expect(attention).toContainText("6 documents need a decision");
    await expect(attention).toContainText("paper-043.pdf");
    await expect(attention.getByRole("button", { name: "Skip file" })).toHaveCount(5);
    await expect(attention).toContainText("1 more in the list below");
    // Every document is one click away, a page at a time, searchable, filterable.
    await expect(page.getByTestId("document-table")).toBeHidden();
    await page.getByText("All 300 documents").click();
    await expect(page.getByTestId("document-table")).toBeVisible();
    const table = page.getByTestId("document-table");
    await expect(table.getByRole("row")).toHaveCount(51);
    await expect(table).toContainText("Showing 50 of 300");
    await table.getByRole("button", { name: /Show 50 more/ }).click();
    await expect(table.getByRole("row")).toHaveCount(101);
    await table.getByLabel("Search documents").fill("paper-29");
    await expect(page.getByTestId("document-table-count")).toContainText("10 of 300 documents");
    await table.getByLabel("Search documents").fill("");
    await table.getByLabel("Phase filter").click();
    await page.getByRole("option", { name: /Needs attention/ }).click();
    await expect(page.getByTestId("document-table-count")).toContainText("6 of 300 documents");
    await expect(table.getByRole("button", { name: "Skip file" })).toHaveCount(6);
  });

  test("selects every shown graph at once, leaving running ones out", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    const bar = page.getByTestId("catalog-selection");
    const rowBoxes = 'input[type="checkbox"][aria-label^="Select "]:not([aria-label="Select all shown graphs"])';
    const rows = page.locator(rowBoxes);
    const deletable = page.locator(`${rowBoxes}:not([disabled])`);
    const running = page.locator(`${rowBoxes}[disabled]`);
    await expect(deletable.first()).toBeVisible();
    const shown = await deletable.count();
    expect(await running.count()).toBeGreaterThan(0);
    await bar.getByLabel("Select all shown graphs").check();
    await expect(bar).toContainText(`${shown} of ${shown} selected`);
    for (const box of await deletable.all()) {
      await expect(box).toBeChecked();
    }
    for (const box of await running.all()) {
      await expect(box).not.toBeChecked();
    }
    // A filter hides some of the selection without dropping it.
    await page.getByLabel("Search graphs").fill("judo");
    await expect(bar).toContainText("not shown");
    await page.getByLabel("Search graphs").fill("");
    // The dialog lists the whole selection; Clear puts it away.
    await page.getByRole("button", { name: "Delete selected" }).click();
    await expect(page.getByRole("dialog")).toContainText(`Delete ${shown} graphs?`);
    await page.getByRole("dialog").getByRole("button", { name: "Cancel" }).click();
    await bar.getByRole("button", { name: "Clear" }).click();
    await expect(bar).toContainText(`Select all ${shown}`);
    expect(await rows.count()).toBeGreaterThan(0);
  });

  test("shows a cancelling leftover-lease sentence", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_cancelling");
    await expect(page.getByTestId("status-sentence")).toContainText("Cancelling");
    await expect(page.getByTestId("status-sentence")).toContainText("leftover leases");
  });

  test("scans a sample pack for PII before embed", async ({ page }) => {
    // The seed writes the pack under the server's state root, which the
    // console keeps to itself; the test knows it from the same environment.
    await page.goto("/?runtime=local&page=new");
    await useFolderPath(page, path.join(STATE_ROOT, "pii-pack"));
    await page.getByRole("button", { name: "Scan for PII" }).click();
    await expect(page.getByText(/PII email/i)).toBeVisible();
    // Acknowledging keeps the finding on screen rather than making it vanish.
    await page.getByRole("button", { name: "Acknowledge residual risk" }).click();
    await expect(page.getByTestId("pii-acknowledged")).toContainText(/PII email .* residual risk acknowledged/);
    await expect(page.getByRole("button", { name: "Acknowledge residual risk" })).toHaveCount(0);
  });

  test("compares estimate vs actual on consumption", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Run details" }).click();
    await expect(page.getByTestId("estimate-vs-actual")).toBeVisible();
  });

  test("proposes an ontology with gold coverage", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Describe the graph you want").fill("people, schools, and techniques in these histories");
    await page.getByRole("button", { name: "Suggest types" }).click();
    const proposal = page.getByTestId("ontology-proposal");
    await expect(proposal).toBeVisible();
    // The review panel marks what the lists do not hold yet, and only that.
    const editor = page.getByTestId("ontology-editor");
    const entityTypes = editor.getByTestId("entity-types");
    await expect(proposal).toContainText("SCHOOL");
    await expect(proposal.getByText("new", { exact: true }).first()).toBeVisible();
    await proposal.getByRole("button", { name: "Add to mine" }).click();
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toBeVisible();
    await expect(proposal.getByText("new", { exact: true })).toHaveCount(0);
    await expect(editor).toContainText("Changed from the default profile.");
    // A description is saved with Enter and abandoned with Cancel.
    await entityTypes.getByRole("button", { name: "SCHOOL", exact: true }).click();
    await entityTypes.getByLabel("Description of SCHOOL").fill("A named place of instruction.");
    await entityTypes.getByLabel("Description of SCHOOL").press("Enter");
    await expect(entityTypes.getByLabel("Description of SCHOOL")).toHaveCount(0);
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toHaveAttribute(
      "title",
      "A named place of instruction.",
    );
    await entityTypes.getByRole("button", { name: "SCHOOL", exact: true }).click();
    await entityTypes.getByRole("button", { name: "Cancel" }).click();
    await expect(entityTypes.getByLabel("Description of SCHOOL")).toHaveCount(0);
    // The panel can be put away without taking anything from it.
    await proposal.getByRole("button", { name: "Dismiss suggestion" }).click();
    await expect(page.getByTestId("ontology-proposal")).toHaveCount(0);
  });

  test("picks neighborhoods from a ranked list inside Filters", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    // The facet lives in the Filters card, not on a row of its own.
    await expect(page.getByRole("button", { name: "Choose neighborhoods" })).toBeHidden();
    const filters = await openFilters(page);
    const facet = filters.getByTestId("facet-neighborhoods");
    // Nothing chosen: one button, no chips, no list.
    await expect(facet.getByRole("button", { name: "Choose neighborhoods" })).toBeVisible();
    await expect(page.getByTestId("facet-neighborhoods-panel")).toHaveCount(0);
    await facet.getByRole("button", { name: "Choose neighborhoods" }).click();
    const panel = page.getByTestId("facet-neighborhoods-panel");
    const options = panel.getByRole("checkbox");
    expect(await options.count()).toBeGreaterThan(1);
    // Largest first, each with its size.
    await expect(options.first()).toHaveAccessibleName(/^TECHNIQUE/);
    await expect(panel.getByRole("listitem").first()).toContainText("28 entities");
    await expect(panel).toContainText("11 neighborhoods · 0 chosen");
    await panel.getByLabel("Find neighborhoods").fill("person");
    await expect(options).toHaveCount(1);
    await expect(panel).toContainText("1 of 11 neighborhoods");
    await panel.getByRole("checkbox", { name: /^PERSON/ }).check();
    await panel.getByLabel("Find neighborhoods").fill("");
    await panel.getByRole("checkbox", { name: /^MARTIAL_ART/ }).check();
    await expect(panel).toContainText("11 neighborhoods · 2 chosen");
    await panel.getByRole("button", { name: "Done" }).click();
    // The chosen ones are chips beside the button; the canvas scope follows them.
    await expect(panel).toHaveCount(0);
    await expect(facet.getByRole("button", { name: "2 chosen" })).toBeVisible();
    await expect(facet.getByRole("button", { name: "Remove PERSON" })).toBeVisible();
    await expect(facet.getByRole("button", { name: "Remove MARTIAL_ART" })).toBeVisible();
    await expect(page.getByTestId("explore-scope")).toContainText("27 of 27 entities · 19 relations");
    await expect(page.getByTestId("filters-summary")).toContainText("2 neighborhoods");
    await facet.getByRole("button", { name: "Remove PERSON" }).click();
    await expect(facet.getByRole("button", { name: "1 chosen" })).toBeVisible();
    await expect(page.getByTestId("filters-summary")).toContainText("1 neighborhood");
    await facet.getByRole("button", { name: "Remove MARTIAL_ART" }).click();
    await expect(facet.getByRole("button", { name: "Choose neighborhoods" })).toBeVisible();
    await expect(page.getByTestId("filters-summary")).toHaveText("Filters");
  });

  test("sorts, searches, expands and pages the explore tables", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Data" }).click();
    // Entities: sorting reorders the whole match before it is paged, and ids
    // sit muted at the end while names lead.
    const entities = page.getByRole("tabpanel", { name: "Entities" });
    await expect(entities.getByTestId("record-table-count")).toHaveText("Showing 50 of 119 entities");
    await expect(entities.getByRole("button", { name: "Show 50 more" })).toBeVisible();
    const firstName = () => entities.getByRole("row").nth(1).getByRole("cell").first();
    await expect(firstName()).toHaveText("Jigoro Kano");
    const nameHeader = entities.getByRole("columnheader", { name: "Name" });
    await expect(nameHeader).toHaveAttribute("aria-sort", "none");
    await nameHeader.getByRole("button").click();
    await expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
    await expect(firstName()).toHaveText("1882");
    await nameHeader.getByRole("button").click();
    await expect(nameHeader).toHaveAttribute("aria-sort", "descending");
    // Sorting ignores case: a lowercase name is not banished to the end.
    await expect(firstName()).toHaveText("wrestling");
    await entities.getByLabel("Find in entities").fill("kano");
    await expect(entities.getByTestId("record-table-count")).toHaveText("1 entity (of 119 total)");
    await expect(entities.getByRole("cell", { name: "jigoro_kano" })).toBeVisible();

    // Relations: 148 rows page 50 at a time, endpoints show names not ids,
    // and confidence sorts as a number.
    await page.getByRole("tab", { name: "Relations" }).click();
    const relations = page.getByRole("tabpanel", { name: "Relations" });
    await expect(relations.getByTestId("record-table-count")).toHaveText("Showing 50 of 148 relations");
    await expect(relations.getByRole("row")).toHaveCount(51);
    await expect(relations.getByRole("row").nth(1).getByRole("cell").first()).toHaveText("Kodokan");
    await expect(relations.getByRole("cell", { name: "rel_001" })).toBeVisible();
    await relations.getByRole("button", { name: "Show 50 more" }).click();
    await expect(relations.getByTestId("record-table-count")).toHaveText("Showing 100 of 148 relations");
    await relations.getByRole("button", { name: "Show 48 more" }).click();
    await expect(relations.getByTestId("record-table-count")).toHaveText("148 relations");
    await expect(relations.getByRole("button", { name: /^Show \d+ more$/ })).toHaveCount(0);
    const confidenceHeader = relations.getByRole("columnheader", { name: "Confidence" });
    await confidenceHeader.getByRole("button").click();
    await expect(confidenceHeader).toHaveAttribute("aria-sort", "ascending");
    await expect(relations.getByRole("row").nth(1).getByRole("cell", { name: "0.6", exact: true })).toBeVisible();
    await confidenceHeader.getByRole("button").click();
    await expect(confidenceHeader).toHaveAttribute("aria-sort", "descending");
    await expect(relations.getByRole("row").nth(1).getByRole("cell", { name: "0.95", exact: true })).toBeVisible();

    // Communities: member counts instead of id lists.
    await page.getByRole("tab", { name: "Neighborhoods" }).click();
    const communities = page.getByRole("tabpanel", { name: "Neighborhoods" });
    await expect(communities.getByTestId("record-table-count")).toHaveText("11 neighborhoods");
    const sizeHeader = communities.getByRole("columnheader", { name: "Size" });
    await sizeHeader.getByRole("button").click();
    await sizeHeader.getByRole("button").click();
    await expect(sizeHeader).toHaveAttribute("aria-sort", "descending");
    await expect(communities.getByRole("row").nth(1).getByRole("cell").first()).toHaveText("TECHNIQUE");
    await expect(communities.getByRole("row").nth(1).getByRole("cell", { name: "28", exact: true })).toBeVisible();

    // Evidence: a long quote is one line until "More" opens it, and search
    // narrows the count against the graph total.
    await page.getByRole("tab", { name: "Evidence" }).click();
    const evidence = page.getByRole("tabpanel", { name: "Evidence" });
    await expect(evidence.getByTestId("record-table-count")).toHaveText("Showing 50 of 161 evidence rows");
    await evidence.getByLabel("Find in evidence rows").fill("governed through the Association");
    await expect(evidence.getByTestId("record-table-count")).toHaveText("1 evidence row (of 161 total)");
    const quote = evidence.getByRole("row").nth(1).getByRole("cell").nth(1);
    await expect(quote).not.toContainText("member commissions");
    await evidence.getByRole("button", { name: "More" }).click();
    await expect(quote).toContainText("and its member commissions.");
    await expect(evidence.getByRole("button", { name: "Less" })).toBeVisible();
    // The quote names the relation it grounds by its endpoints, not by id.
    await expect(evidence.getByRole("row").nth(1).getByRole("cell").nth(3)).toContainText("GOVERNED_BY");

    // A graph filter narrows every tab, and each count line says what the
    // graph has in total so the narrowing is visible.
    await evidence.getByLabel("Find in evidence rows").fill("");
    await page.getByRole("tab", { name: "Explore" }).click();
    await page.getByPlaceholder("Entity name or description").fill("judo");
    await page.getByRole("tab", { name: "Data" }).click();
    await page.getByRole("tab", { name: "Evidence" }).click();
    await expect(evidence.getByTestId("record-table-count")).toHaveText("1 evidence row (of 161 total)");
    await expect(evidence.getByRole("row").nth(1).getByRole("cell").nth(3)).toContainText("Judo GOVERNED_BY International Judo Federation");
    await page.getByRole("tab", { name: "Neighborhoods" }).click();
    await expect(communities.getByTestId("record-table-count")).toHaveText(/^\d+ neighborhoods \(of 11 total\)$/);
    await expect(communities.getByRole("cell", { name: "MARTIAL_ART", exact: true })).toBeVisible();
    await expect(communities.getByRole("cell", { name: "PERSON", exact: true })).toHaveCount(0);
    await page.getByRole("tab", { name: "Entities" }).click();
    await expect(entities.getByTestId("record-table-count")).toHaveText("3 entities (of 119 total)");
  });

  test("jumps from a Data row to the graph and back", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Data" }).click();
    const entities = page.getByRole("tabpanel", { name: "Entities" });
    await entities.getByLabel("Find in entities").fill("Judo");
    await entities.getByRole("button", { name: "Show in graph: Judo" }).click();
    await expect(page.getByRole("tab", { name: "Explore" })).toHaveAttribute("data-state", "active");
    await expect(page.getByTestId("explore-selection")).toContainText("Judo");
    // A neighborhood row focuses the canvas on it, which the Data tab then reports.
    await page.getByRole("tab", { name: "Data" }).click();
    await page.getByRole("tab", { name: "Neighborhoods" }).click();
    await page.getByRole("button", { name: "Focus in graph: PERSON" }).click();
    await expect(page.getByRole("tab", { name: "Explore" })).toHaveAttribute("data-state", "active");
    await expect(page.getByTestId("filters-count")).toHaveText("1");
    await page.getByRole("tab", { name: "Data" }).click();
    await expect(page.getByTestId("graph-data-scope")).toContainText("1 neighborhood");
    await page.getByRole("tab", { name: "Entities" }).click();
    await expect(page.getByRole("tabpanel", { name: "Entities" }).getByTestId("record-table-count")).toContainText("(of 119 total)");
  });

  test("shows a 1-hop neighborhood from a selected relation", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Data" }).click();
    await page.getByRole("tab", { name: "Relations" }).click();
    // A row leads back to the canvas: the Explore tab opens with it selected.
    await page.getByRole("cell", { name: "rel_001" }).click();
    await expect(page.getByRole("tab", { name: "Explore" })).toHaveAttribute("data-state", "active");
    const selection = page.getByTestId("explore-selection");
    await expect(selection.getByRole("button", { name: "Looks wrong" })).toBeVisible();
    await selection.getByRole("button", { name: "Show neighborhood" }).click();
    await expect(page.getByTestId("explore-scope")).toContainText("in the selected neighborhood");
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
  });

  test("welcome back ignores the seed OCR fixture after ALICE signs in", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign in as ALICE" }).click();
    await expect(page.getByTestId("welcome-back")).toBeVisible();
    await expect(page.getByTestId("welcome-back")).not.toContainText("Active OCR");
  });
});

test.describe("request confinement", () => {
  test("refuses ids and paths that would name a directory elsewhere", async ({ request }) => {
    const escaped = await request.get("/api/inspect?run=../../etc");
    expect(escaped.status()).toBe(400);
    const cancel = await request.post("/api/trpc/runs.cancel", {
      headers: { "x-flakegraph-runtime": "local" },
      data: { json: { runId: "../../x" } },
    });
    expect(cancel.status()).toBe(400);
    const listed = await request.get(
      `/api/trpc/ingestion.sources.list?input=${encodeURIComponent(JSON.stringify({ json: { source: { kind: "local", path: "/" } } }))}`,
      { headers: { "x-flakegraph-runtime": "local" } },
    );
    expect(listed.status()).toBe(400);
    expect(await listed.text()).toContain("source.path must be inside");
    const upload = await request.post("/api/uploads", { multipart: { files: { name: "note.md", mimeType: "text/markdown", buffer: Buffer.from("x") }, jobId: "../../../tmp/x" } });
    expect(upload.status()).toBe(400);
  });

  test("keeps a bearer key acting as the principal that minted it", async ({ page }) => {
    await page.goto("/?runtime=local&page=keys");
    await signInAs(page, "ALICE");
    await page.keyboard.press("Escape");
    await page.getByLabel("Key name").fill("owned-by-alice");
    await page.getByRole("button", { name: "Create a key" }).click();
    const secretLine = await page.getByText(/Copy now:/).innerText();
    const secret = secretLine.replace(/^.*Copy now:\s*/, "").trim();
    const session = await page.request.get(
      "/api/trpc/auth.session?batch=1&input=%7B%220%22%3A%7B%22json%22%3Anull%7D%7D",
      { headers: { Authorization: `Bearer ${secret}`, "x-flakegraph-runtime": "local" } },
    );
    expect(session.ok()).toBeTruthy();
    const [{ result }] = (await session.json()) as [{ result: { data: { json: { viewer: { userName: string }; authorization: string } } } }];
    expect(result.data.json.viewer.userName).toBe("ALICE");
    expect(result.data.json.authorization).toBe("machine");
    await page.getByLabel("Search keys").fill("owned-by-alice");
    await page.getByRole("button", { name: "Revoke owned-by-alice" }).click();
    await page.getByRole("button", { name: "Confirm revoke owned-by-alice" }).click();
    await expect(page.getByTestId("sdk-key-list")).toContainText("No key matches this search.");
  });
});

test.describe("health", () => {
  test("serves the kubernetes health route", async ({ request }) => {
    const response = await request.get("/api/health");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.service).toBe("flakegraph-control-plane");
  });
});
