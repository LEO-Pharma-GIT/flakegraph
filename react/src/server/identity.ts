import type { Viewer } from "./protocol/schema";

/**
 * The headers the sign-in gate (oauth2-proxy behind Traefik forwardAuth)
 * writes on a request it admitted, and nothing else. The gate overwrites
 * exactly these on every request, so a value a browser sent itself never
 * survives; any other header - X-Forwarded-*, preferred-username - is not
 * overwritten and would be whatever the caller chose.
 *
 * The address comes first because it is what a person recognises as
 * themselves; the gate's "user" is often the provider's opaque subject id.
 */
export const GATE_IDENTITY_HEADERS = {
  email: "x-auth-request-email",
  user: "x-auth-request-user",
  groups: "x-auth-request-groups",
} as const;

export function unidentifiedViewer(): Viewer {
  return { userName: "", email: "", roles: [] };
}

export function viewerFromHeaders(
  headers: Headers,
  options: { trustIdentityHeaders: boolean; snowflakeHosted: boolean },
): Viewer {
  if (options.snowflakeHosted) {
    // Snowflake's ingress overwrites this header and writes no other that names
    // the viewer, so the roles are asked of Snowflake by the runtime: a
    // roles or email header here is whatever the caller chose to send.
    const tokenUser = headers.get("sf-context-current-user");
    if (tokenUser?.trim()) {
      return { userName: tokenUser.trim().toUpperCase(), email: "", roles: [] };
    }
  }
  if (!options.trustIdentityHeaders) {
    return unidentifiedViewer();
  }
  const email = headers.get(GATE_IDENTITY_HEADERS.email)?.trim() ?? "";
  const name = email || headers.get(GATE_IDENTITY_HEADERS.user)?.trim() || "";
  if (!name) {
    return unidentifiedViewer();
  }
  return {
    userName: name.toUpperCase(),
    email,
    roles: parseRoles(headers.get(GATE_IDENTITY_HEADERS.groups)),
  };
}

/**
 * The viewer's own Snowflake token for this request, present when the app is
 * granted caller's rights. Joined to the service token, it opens a session that
 * answers as the viewer.
 */
export function callerTokenFromHeaders(headers: Headers): string | null {
  return headers.get("sf-context-current-user-token")?.trim() || null;
}

function parseRoles(value: string | null): string[] {
  if (!value) {
    return [];
  }
  return value
    .split(/[,\s]+/)
    .map((role) => role.trim().toUpperCase())
    .filter(Boolean);
}
