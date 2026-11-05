"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function ErrorPage({
  error,
  reset,
}: {
  error: unknown;
  reset: () => void;
}) {
  const message = error instanceof Error && typeof error.message === "string" ? error.message : null;
  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-3 p-8">
      <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Control plane</p>
      <h1 className="text-2xl font-semibold tracking-tight">Something went wrong</h1>
      <p className="max-w-md text-center text-sm text-muted-foreground">
        The control plane hit an unexpected error. Try again, or go back to the graph catalog.
      </p>
      {message ? (
        <details className="max-w-md text-center text-xs text-muted-foreground">
          <summary className="cursor-pointer">Technical details</summary>
          <p className="mt-2 break-words">{message}</p>
        </details>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button onClick={reset}>Try again</Button>
        <Button variant="outline" asChild>
          <Link href="/">Back to graphs</Link>
        </Button>
      </div>
    </main>
  );
}
