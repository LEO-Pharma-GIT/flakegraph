import { defineConfig } from "drizzle-kit";

// drizzle-kit introspects the coordination store the fleet writes; it must
// be told which one rather than guess at a local default with a password
// baked into the repository.
const url = process.env.DATABASE_URL?.trim();
if (!url) {
  throw new Error("Set DATABASE_URL to the coordination Postgres before running drizzle-kit.");
}

export default defineConfig({
  schema: "./src/server/db/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: { url },
  introspect: {
    casing: "preserve",
  },
});
