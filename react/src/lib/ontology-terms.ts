/** How a name must read to be a type: PERSON, FOUNDED_BY. */
const TERM_NAME = /^[A-Z][A-Z0-9_]{1,31}$/;

/** Turn what someone typed into a type name, or nothing. */
export function normalizeTermName(raw: string): string | null {
  const name = raw
    .trim()
    .replace(/[\s-]+/g, "_")
    .replace(/[^A-Za-z0-9_]/g, "")
    .toUpperCase();
  return TERM_NAME.test(name) ? name : null;
}
