import type { SnowflakeSql, SqlBind, SqlRow } from "./client";

type Response = SqlRow[] | Error | ((binds: readonly SqlBind[]) => SqlRow[]);

/**
 * A scripted Snowflake for tests: each statement is answered by the first
 * response whose fragment it contains, and every statement is recorded so a
 * test can assert on what reached SQL.
 */
export class RecordingSql implements SnowflakeSql {
  readonly statements: Array<{ sql: string; binds: readonly SqlBind[] }> = [];

  constructor(private readonly responses: Array<[fragment: string, response: Response]> = []) {}

  get texts(): string[] {
    return this.statements.map((statement) => statement.sql);
  }

  async query(sql: string, binds: readonly SqlBind[] = []): Promise<SqlRow[]> {
    this.statements.push({ sql, binds });
    const match = this.responses.find(([fragment]) => sql.includes(fragment));
    if (!match) {
      return [];
    }
    const response = match[1];
    if (response instanceof Error) {
      throw response;
    }
    return typeof response === "function" ? response(binds) : response;
  }

  async *stream(sql: string, binds: readonly SqlBind[] = []): AsyncIterable<SqlRow> {
    yield* await this.query(sql, binds);
  }
}
