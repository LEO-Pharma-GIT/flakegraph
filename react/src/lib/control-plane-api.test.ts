import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { identifiedProcedure, operatorProcedure, protectedProcedure, sessionProcedure } from "@/server/trpc/init";
import { appRouter } from "@/server/trpc/router";
import {
  API_AREAS,
  CONTROL_PLANE_ENDPOINTS,
  CONTROL_PLANE_PROCEDURES,
  CONTROL_PLANE_ROUTES,
  askableGraphsExamples,
  filterEndpoints,
  inspectReportExamples,
  type ApiAccess,
} from "./control-plane-api";

interface ProcedureDef {
  type: "query" | "mutation" | "subscription";
  middlewares: unknown[];
}

/** The router's procedures by dotted path, as the HTTP adapter resolves them. */
const procedures = (appRouter._def as unknown as { procedures: Record<string, { _def: ProcedureDef }> }).procedures;

/** A builder's own middleware: the one it adds to what it was built from. */
function ownMiddleware(builder: unknown): unknown {
  const middlewares = (builder as { _def: { middlewares: unknown[] } })._def.middlewares;
  return middlewares[middlewares.length - 1];
}

/** The access a procedure was built with, read from the middlewares it carries. */
function builtAccess(definition: ProcedureDef): ApiAccess {
  if (definition.middlewares.includes(ownMiddleware(operatorProcedure))) {
    return "operator";
  }
  if (definition.middlewares.includes(ownMiddleware(sessionProcedure))) {
    return "session";
  }
  if (definition.middlewares.includes(ownMiddleware(identifiedProcedure))) {
    return "named";
  }
  return definition.middlewares.includes(ownMiddleware(protectedProcedure)) ? "write" : "read";
}

const API_ROOT = path.resolve(__dirname, "../app/api");
const METHODS = "GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS";

/** A dynamic segment, whatever it is called: `[trpc]` in a directory, `{procedure}` in the catalog. */
const anySegment = (routePath: string) => routePath.replace(/\[[^\]]+\]|\{[^}]+\}/g, "{}");

/** Every `route.ts` under app/api, as `METHOD /path`. */
function servedRoutes(): string[] {
  const served: string[] = [];
  const visit = (directory: string) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(full);
        continue;
      }
      if (entry.name !== "route.ts") {
        continue;
      }
      const source = readFileSync(full, "utf8");
      const relative = path.relative(API_ROOT, directory).split(path.sep).join("/");
      const routePath = anySegment(`/api/${relative}`);
      const declared = [...source.matchAll(new RegExp(`export\\s+(?:async\\s+)?function\\s+(${METHODS})\\b`, "g"))];
      const aliased = [...source.matchAll(/export\s*\{([^}]*)\}/g)].flatMap((match) =>
        [...match[1].matchAll(new RegExp(`\\bas\\s+(${METHODS})\\b`, "g"))],
      );
      for (const method of [...declared, ...aliased].map((match) => match[1])) {
        served.push(`${method} ${routePath}`);
      }
    }
  };
  visit(API_ROOT);
  return served.sort();
}

describe("the API catalog", () => {
  it("lists exactly the router's procedures, with their kind and area", () => {
    const routed = Object.entries(procedures)
      .map(([name, procedure]) => `${name} ${procedure._def.type} ${name.split(".")[0]}`)
      .sort();
    const listed = CONTROL_PLANE_PROCEDURES.map((entry) => `${entry.name} ${entry.kind} ${entry.area}`).sort();
    expect(listed).toEqual(routed);
  });

  it("gives each procedure the access level it was built with", () => {
    for (const entry of CONTROL_PLANE_PROCEDURES) {
      expect({ name: entry.name, access: entry.access }).toEqual({
        name: entry.name,
        access: builtAccess(procedures[entry.name]._def),
      });
    }
  });

  it("lists exactly the methods each route under app/api answers", () => {
    const listed = CONTROL_PLANE_ROUTES.map((entry) => `${entry.kind} ${anySegment(entry.name)}`).sort();
    expect(listed).toEqual(servedRoutes());
  });

  it("names each endpoint once, in a known area, with something to say about it", () => {
    const keys = CONTROL_PLANE_ENDPOINTS.map((entry) => `${entry.kind} ${entry.name}`);
    expect(new Set(keys).size).toBe(keys.length);
    const areas = new Set<string>(API_AREAS.map((area) => area.id));
    for (const entry of CONTROL_PLANE_ENDPOINTS) {
      expect(areas.has(entry.area), entry.name).toBe(true);
      expect(entry.purpose.trim(), entry.name).not.toBe("");
      expect(entry.input.trim(), entry.name).not.toBe("");
      expect(entry.returns.trim(), entry.name).not.toBe("");
    }
  });

  it("narrows by every word of a search, across name, access and input", () => {
    expect(filterEndpoints(CONTROL_PLANE_ENDPOINTS, "")).toHaveLength(CONTROL_PLANE_ENDPOINTS.length);
    expect(filterEndpoints(CONTROL_PLANE_ENDPOINTS, "gold.upload").map((entry) => entry.name)).toEqual(["graphs.gold.upload"]);
    const named = filterEndpoints(CONTROL_PLANE_ENDPOINTS, "named person").map((entry) => entry.name);
    expect(named).toEqual(expect.arrayContaining(["graphs.share"]));
    const session = filterEndpoints(CONTROL_PLANE_ENDPOINTS, "browser session").map((entry) => entry.name);
    expect(session).toEqual(expect.arrayContaining(["workspace.createKey", "workspace.revokeKey"]));
    const operator = filterEndpoints(CONTROL_PLANE_ENDPOINTS, "operator").map((entry) => entry.name);
    expect(operator).toEqual(expect.arrayContaining(["auth.setSuggestions", "workspace.markGrant"]));
    expect(filterEndpoints(CONTROL_PLANE_ENDPOINTS, "multipart").map((entry) => `${entry.kind} ${entry.name}`)).toEqual([
      "POST /api/uploads",
    ]);
  });
});

describe("REST examples", () => {
  const origin = "https://console.example.test";

  it("lists askable graphs with a key in every language", () => {
    for (const code of Object.values(askableGraphsExamples(origin))) {
      expect(code).toContain(origin);
      expect(code).toContain("/api/graphs");
      expect(code).toContain("FLAKEGRAPH_API_KEY");
    }
  });

  it("fetches the inspect report by run in every language", () => {
    for (const code of Object.values(inspectReportExamples(origin))) {
      expect(code).toContain(origin);
      expect(code).toContain("/api/inspect?");
      expect(code).toContain("run_martial_arts");
      expect(code).toContain("FLAKEGRAPH_API_KEY");
    }
  });
});
