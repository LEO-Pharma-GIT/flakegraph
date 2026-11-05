// Deploys the console to Snowflake App Runtime.
//
// The remote builder uploads one directory. The console needs the run
// configuration defaults and ontology profiles from the repository's configs/,
// which sit beside react/ rather than in it, and uploading the repository root
// would ship the whole Python tree. So this stages the console and those
// configs together in a temporary directory and deploys from there.
//
// app.yml stays account-neutral. These variables name the account's objects:
//   FLAKEGRAPH_SNOWFLAKE_APP        application service name
//   FLAKEGRAPH_SNOWFLAKE_DATABASE   database holding the service and the KG_* tables
//   FLAKEGRAPH_SNOWFLAKE_SCHEMA     schema holding the service and the KG_* tables
//   FLAKEGRAPH_SNOWFLAKE_WAREHOUSE  warehouse for the console's queries
//   FLAKEGRAPH_SNOWFLAKE_OPERATORS  users and roles who administer the console,
//                                   comma-separated; FLAKEGRAPH_APP_OPERATORS on the service
//
// Usage: bun scripts/deploy-snowflake.ts [snow app deploy options, e.g. -c <connection>]
import { spawnSync } from "node:child_process";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseDocument } from "yaml";

const consoleRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(consoleRoot, "..");

// Local-only paths; app.yml lists the same ones so a direct upload skips them too.
const EXCLUDED = new Set([
  "node_modules",
  ".next",
  "test-results",
  "playwright-report",
  "coverage",
  "e2e",
]);

async function main() {
  const stage = await mkdtemp(path.join(tmpdir(), "flakegraph-app-"));
  try {
    await cp(consoleRoot, stage, {
      recursive: true,
      filter: (source) => {
        const name = path.basename(source);
        return !EXCLUDED.has(name) && !name.startsWith(".env") && !name.endsWith(".tsbuildinfo");
      },
    });
    // The console finds its repository root by walking up to
    // configs/app-defaults.yaml, so the deployed bundle carries it at its root.
    await cp(path.join(repositoryRoot, "configs", "app-defaults.yaml"), path.join(stage, "configs", "app-defaults.yaml"));
    await cp(path.join(repositoryRoot, "configs", "ontologies"), path.join(stage, "configs", "ontologies"), {
      recursive: true,
    });
    await writeFile(path.join(stage, "app.yml"), await accountAppYml(path.join(consoleRoot, "app.yml")));

    const deploy = spawnSync("snow", ["app", "deploy", ...process.argv.slice(2)], {
      cwd: stage,
      stdio: "inherit",
    });
    process.exitCode = deploy.status ?? 1;
  } finally {
    await rm(stage, { recursive: true, force: true });
  }
}

/** Returns app.yml with this account's names in place of the neutral defaults. */
async function accountAppYml(file: string): Promise<string> {
  const document = parseDocument(await readFile(file, "utf8"));
  const overrides: Array<[string, string | undefined]> = [
    ["name", process.env.FLAKEGRAPH_SNOWFLAKE_APP],
    ["database", process.env.FLAKEGRAPH_SNOWFLAKE_DATABASE],
    ["schema", process.env.FLAKEGRAPH_SNOWFLAKE_SCHEMA],
    ["query_warehouse", process.env.FLAKEGRAPH_SNOWFLAKE_WAREHOUSE],
  ];
  for (const [key, value] of overrides) {
    if (value?.trim()) {
      document.set(key, value.trim());
    }
  }
  const warehouse = process.env.FLAKEGRAPH_SNOWFLAKE_WAREHOUSE?.trim();
  if (warehouse) {
    const variables = document.get("environment_variables", true) as { items: Array<{ get(key: string): unknown; set(key: string, value: string): void }> };
    for (const variable of variables.items) {
      if (variable.get("name") === "FLAKEGRAPH_SNOWFLAKE_WAREHOUSE") {
        variable.set("value", warehouse);
      }
    }
  }
  // Only ever named here: who administers a console is the account's business.
  const operators = process.env.FLAKEGRAPH_SNOWFLAKE_OPERATORS?.trim();
  if (operators) {
    document.addIn(["environment_variables"], { name: "FLAKEGRAPH_APP_OPERATORS", value: operators });
  }
  return document.toString();
}

await main();
