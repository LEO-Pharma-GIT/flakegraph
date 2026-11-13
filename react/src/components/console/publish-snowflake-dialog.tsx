"use client";

import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

/** The Snowflake coordinates a fleet's processing profile carries, when it has an account at all. */
export interface FleetSnowflake {
  account: string;
  user: string;
  database: string;
  schema: string;
  warehouse: string;
  role: string;
  bulkStage: string;
}

export function fleetSnowflakeFromProfile(profile: { config: Record<string, unknown> } | null | undefined): FleetSnowflake | null {
  const values = profile?.config.snowflake;
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    return null;
  }
  const record = values as Record<string, unknown>;
  const text = (key: string) => (typeof record[key] === "string" && !String(record[key]).startsWith("${") ? String(record[key]) : "");
  if (!text("account") || !text("user")) {
    return null;
  }
  return {
    account: text("account"),
    user: text("user"),
    database: text("database"),
    schema: text("schema"),
    warehouse: text("warehouse"),
    role: text("role"),
    bulkStage: text("bulk_stage"),
  };
}

/**
 * Write a finished graph into Snowflake as it is. The account and credential
 * are the fleet's; the person names where inside it the graph lands, with
 * the profile's defaults already in the fields.
 */
export function PublishSnowflakeDialog({
  open,
  onOpenChange,
  runId,
  graphName,
  fleet,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  runId: string;
  graphName: string;
  fleet: FleetSnowflake;
}) {
  const [database, setDatabase] = useState(fleet.database);
  const [schema, setSchema] = useState(fleet.schema);
  const [warehouse, setWarehouse] = useState(fleet.warehouse);
  const [role, setRole] = useState(fleet.role);
  const [bulkStage, setBulkStage] = useState(fleet.bulkStage);
  const publish = trpc.graphs.publishSnowflake.useMutation({
    onSuccess: (result) => {
      toast.success(`Published ${result.nodes.toLocaleString()} entities and ${result.edges.toLocaleString()} relations to ${database}.${schema}.`);
      onOpenChange(false);
    },
    onError: (error) => toast.error(error.message),
  });
  const ready = database.trim() && schema.trim() && warehouse.trim() && bulkStage.trim();
  return (
    <Dialog open={open} onOpenChange={(next) => (publish.isPending ? undefined : onOpenChange(next))}>
      <DialogContent aria-describedby="publish-snowflake-description">
        <DialogHeader>
          <DialogTitle>Publish to Snowflake</DialogTitle>
          <DialogDescription id="publish-snowflake-description">
            Writes “{graphName}” as it is into the fleet&apos;s Snowflake account ({fleet.account}, as {fleet.user}). Nothing
            is parsed or extracted again; the graph&apos;s tables are bulk-loaded through the stage below.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Database" value={database} onChange={setDatabase} />
          <Field label="Schema" value={schema} onChange={setSchema} />
          <Field label="Warehouse" value={warehouse} onChange={setWarehouse} />
          <Field label="Role" value={role} onChange={setRole} optional />
          <Field label="Bulk stage" value={bulkStage} onChange={setBulkStage} placeholder="FLAKEGRAPH_BULK" />
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="outline" disabled={publish.isPending} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            disabled={!ready || publish.isPending}
            onClick={() =>
              publish.mutate({
                runId,
                database: database.trim(),
                schema: schema.trim(),
                warehouse: warehouse.trim(),
                role: role.trim() || undefined,
                bulkStage: bulkStage.trim(),
              })
            }
          >
            {publish.isPending ? "Publishing…" : "Publish"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  value,
  onChange,
  optional = false,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  optional?: boolean;
  placeholder?: string;
}) {
  return (
    <label className="grid gap-1 text-sm">
      <span className="font-medium">
        {label}
        {optional ? <span className="ml-1 font-normal text-muted-foreground">optional</span> : null}
      </span>
      <Input aria-label={label} value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}
