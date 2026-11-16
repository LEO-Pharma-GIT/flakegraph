import "server-only";
import { appEnv, parseRuntime } from "../env";
import type { RuntimeMode, Viewer } from "../protocol/schema";
import type { ControlPlane, GraphAccessPlane } from "../protocol/runtime";
import { controlledPlane } from "../access/controlled-plane";
import { unidentifiedViewer } from "../identity";
import { createKubernetesRuntime } from "./kubernetes";
import { createLocalRuntime } from "./local";
import { createSnowflakePlane } from "./snowflake";
import { createSnowflakeStubRuntime } from "./snowflake-stub";

/**
 * The runtime a request is served by, seen through its viewer's access.
 *
 * Every procedure reaches graphs through this, so every read is limited to
 * the viewer's own graphs and those shared with them. A deployment that
 * requires sign-in treats a graph nobody owns as nobody's.
 *
 * Inside Snowflake the account's own tables hold who owns what, so the runtime
 * reading them is the plane: there is no console store to wrap it with.
 * `callerToken` is the viewer's Snowflake token, used only there.
 */
export function resolveRuntime(runtime: RuntimeMode, viewer?: Viewer, callerToken?: string | null): GraphAccessPlane {
  const env = appEnv();
  if (env.snowflakeHosted && !env.stubRuntimes) {
    return createSnowflakePlane(viewer, callerToken);
  }
  return controlledPlane(runtimeFor(runtime, viewer), viewer ?? unidentifiedViewer(), env.stateRoot, {
    strict: env.requireSignIn,
    laptop: !env.requireSignIn && env.demoIdentities,
  });
}

function runtimeFor(runtime: RuntimeMode, viewer?: Viewer): ControlPlane {
  const env = appEnv();
  const selected = env.snowflakeHosted ? "snowflake" : runtime;
  if (selected === "kubernetes") {
    return createKubernetesRuntime(viewer ?? null);
  }
  if (selected === "snowflake") {
    return createSnowflakeStubRuntime(viewer);
  }
  return createLocalRuntime(viewer ?? null);
}

export function runtimeFromHeader(header: string | null): RuntimeMode {
  const env = appEnv();
  if (env.snowflakeHosted) {
    return "snowflake";
  }
  return parseRuntime(header ?? undefined, env.defaultRuntime);
}

export function availableRuntimes(): RuntimeMode[] {
  if (appEnv().snowflakeHosted) {
    return ["snowflake"];
  }
  return ["local", "kubernetes", "snowflake"];
}
