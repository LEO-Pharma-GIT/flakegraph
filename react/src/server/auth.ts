import { matchApiKey, presentedApiSecret, type PresentedCredential } from "./api-keys";
import { appEnv } from "./env";
import { unidentifiedViewer, viewerFromHeaders } from "./identity";
import type { Viewer } from "./protocol/schema";
import { loadWorkspace, type ApiKey, type WorkspaceState } from "./workspace";

/** A request the control plane refuses to serve, and the status that says so. */
export class RequestRefused extends Error {
  readonly status = 401;
}

/**
 * How a request came to be served.
 *
 * - `machine`: it presented an API key the workspace holds.
 * - `gate`: a trusted sign-in gate wrote who the viewer is.
 * - `operator`: an isolated deployment pins one operator in server configuration.
 * - `demo`: the console runs with demo identities, a laptop or rehearsal
 *   device with no gate; the viewer is whoever was assumed, or nobody.
 * - `anonymous`: nobody vouched for the caller; only read-only queries are
 *   served, and only where the deployment opened them.
 */
export type AuthorizationMode = "machine" | "gate" | "operator" | "demo" | "anonymous";

export interface Authorized {
  viewer: Viewer;
  mode: AuthorizationMode;
  /** The machine key the request presented, when it did; a browser session presents none. */
  machine: ApiKey | null;
  /** True when the viewer was assumed from the workspace, a rehearsal device for an ungated console. */
  assumed: boolean;
}

export const SIGN_IN_HINT =
  "Sign in through the gate, or present an API key. " +
  "A console with no gate serves demo identities only when FLAKEGRAPH_DEMO_IDENTITIES=1.";

/**
 * Decide who a request is from, or refuse it.
 *
 * A machine presents an API key and is served as the principal that minted
 * the key; a key that matches nothing is refused. A browser is whoever the
 * gate says it is, where a gate is trusted. With demo identities the viewer
 * is the workspace's assumed identity, and a session that assumed nobody is
 * still served: the device is a laptop with no gate. Everyone else is
 * anonymous, and an anonymous caller is refused outright unless the
 * deployment opened read-only queries to them.
 */
export async function authorizeRequest(headers: Headers, workspace?: WorkspaceState): Promise<Authorized> {
  const env = appEnv();
  const state = workspace ?? (await loadWorkspace(env.stateRoot));
  const presentedSecret = presentedApiSecret(headers);
  if (presentedSecret) {
    const key = matchApiKey(state.apiKeys, presentedSecret);
    if (!key) {
      throw new RequestRefused("API key missing or revoked. This path does not use SSO.");
    }
    return { viewer: keyPrincipal(key), mode: "machine", machine: key, assumed: false };
  }
  const headerViewer = viewerFromHeaders(headers, {
    trustIdentityHeaders: env.trustIdentityHeaders,
    snowflakeHosted: env.snowflakeHosted,
  });
  if (headerViewer.userName) {
    return { viewer: headerViewer, mode: "gate", machine: null, assumed: false };
  }
  if (env.requireSignIn) {
    throw new RequestRefused(SIGN_IN_HINT);
  }
  if (env.singleOperator) {
    return {
      viewer: { userName: env.singleOperator, email: env.singleOperator, roles: [] },
      mode: "operator", machine: null, assumed: false,
    };
  }
  // A gate that is trusted outranks the workspace's own answer: an assumed
  // identity is a rehearsal device, never an override of the gate.
  if (env.demoIdentities && !env.trustIdentityHeaders) {
    const assumed = state.identity;
    return { viewer: assumed ?? unidentifiedViewer(), mode: "demo", machine: null, assumed: Boolean(assumed) };
  }
  if (!env.anonymousRead) {
    throw new RequestRefused(SIGN_IN_HINT);
  }
  return { viewer: unidentifiedViewer(), mode: "anonymous", machine: null, assumed: false };
}

/** Whether the request may change anything or name a path on this host. */
export function mayMutate(authorized: Pick<Authorized, "mode">): boolean {
  return authorized.mode !== "anonymous";
}

/**
 * Whether the request is from someone who administers this console: who may
 * change what every session sees - the suggestion mode, the Snowflake grants
 * checklist, the incident note, a demo console's assumed identity.
 *
 * A program never is. A request that presented a key or an access token is
 * refused whoever it names, so a leaked credential cannot change the console
 * for everyone. A console one person runs - a pinned single operator, or a
 * demo laptop with no gate, where whoever reaches it may already assume any
 * identity - is that person's. A console the gate or Snowflake serves to many
 * people is administered by the principals and roles FLAKEGRAPH_APP_OPERATORS
 * names, and by nobody while it names none.
 */
export function administersConsole(
  authorized: Pick<Authorized, "mode" | "viewer">,
  credential: PresentedCredential | null,
): boolean {
  if (credential) {
    return false;
  }
  switch (authorized.mode) {
    case "operator":
    case "demo":
      return true;
    case "gate": {
      const operators = new Set(appEnv().operators);
      const { userName, email, roles } = authorized.viewer;
      return [userName, email, ...roles].some((name) => Boolean(name.trim()) && operators.has(name.trim().toUpperCase()));
    }
    default:
      return false;
  }
}

/** The 401 a route handler answers with when `authorizeRequest` refused. */
export function refusal(error: unknown): Response | null {
  if (error instanceof RequestRefused) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return null;
}

/**
 * A key acts as the principal that minted it, so what a machine submits is
 * owned by someone. A key with no recorded owner acts as nobody.
 */
function keyPrincipal(key: ApiKey): Viewer {
  const owner = key.owner?.trim();
  return owner ? { userName: owner.toUpperCase(), email: "", roles: [] } : unidentifiedViewer();
}
