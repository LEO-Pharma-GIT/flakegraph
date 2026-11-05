import type { Viewer } from "../protocol/schema";
import { UPLOAD_NAMESPACE } from "../snowflake/stage";
import type { AccessPolicy } from "./graph-access";

/**
 * How `roleFor` reads a Snowflake graph. The tables keep graphs no console ever
 * claimed, run from the CLI or by another deployment, and those stay open to
 * everyone Snowflake signs in; a Snowflake user name is also not an address.
 * Sign-in itself is the ingress's, so this is never the strict policy.
 */
export const SNOWFLAKE_ACCESS_POLICY: AccessPolicy = { strict: false };

/**
 * The rule `roleFor` applies, as a condition on one row of KG_GRAPH.
 *
 * The console's session holds the service owner's rights, so a query returns
 * every row that role can read and this condition is what stands between one
 * viewer's corpus and another's. It runs inside the query that reads the rows,
 * so a list is cut to its size after other people's runs are gone. Identities
 * are bound, never inlined: a user name comes from the identity provider.
 */
export function visibleGraphPredicate(viewer: Viewer, alias = "G"): { sql: string; parameters: string[] } {
  if (!viewer.userName) {
    return { sql: `${alias}.OWNER IS NULL`, parameters: [] };
  }
  const roles = [...viewer.roles].sort();
  const parameters = [viewer.userName, viewer.userName];
  let roleArm = "";
  if (roles.length > 0) {
    const placeholders = roles.map(() => "?").join(", ");
    roleArm = ` OR (UPPER(A.GRANTEE_TYPE) = 'ROLE' AND UPPER(A.GRANTEE) IN (${placeholders}))`;
    parameters.push(...roles);
  }
  const shared =
    `EXISTS (SELECT 1 FROM KG_GRAPH_ACL A WHERE A.GRAPH_ID = ${alias}.GRAPH_ID ` +
    `AND ((UPPER(A.GRANTEE_TYPE) = 'USER' AND UPPER(A.GRANTEE) = ?)${roleArm}))`;
  return {
    sql: `(${alias}.OWNER IS NULL OR UPPER(${alias}.OWNER) = ? OR ${shared})`,
    parameters,
  };
}

/** Where one viewer's upload for a run lands in a stage: theirs by its path. */
export function viewerStagePrefix(viewer: Viewer, jobId: string): string {
  return `${UPLOAD_NAMESPACE}/${sanitizePrincipal(viewer.userName || "shared")}/${jobId}`;
}

/**
 * Whether a viewer may read a path in a stage. The console lists stages with
 * its own rights, so the resolved path is what is authorised: everything
 * outside the upload namespace is a curated corpus anyone may run on, and
 * inside it only the viewer's own folder is theirs.
 */
export function mayReadStagePath(viewer: Viewer, stagePath: string): boolean {
  const segments = stagePath.trim().split("/").filter((segment) => segment && segment !== ".");
  if (segments.length === 0) {
    return false;
  }
  if (segments[0] !== UPLOAD_NAMESPACE) {
    return true;
  }
  const mine = viewerStagePrefix(viewer, "").replace(/\/+$/, "").split("/");
  return mine.every((segment, index) => segments[index] === segment);
}

function sanitizePrincipal(value: string): string {
  const cleaned = value.replace(/[^A-Za-z0-9._-]/g, "_").replace(/^_+|_+$/g, "");
  if (!cleaned || [...cleaned].every((character) => character === ".")) {
    return "shared";
  }
  return cleaned;
}
