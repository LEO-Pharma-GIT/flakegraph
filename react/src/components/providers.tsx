"use client";

import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { httpBatchLink } from "@trpc/client";
import { createTRPCReact } from "@trpc/react-query";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { useState, type ReactNode } from "react";
import { toast } from "sonner";
import superjson from "superjson";
import type { AppRouter } from "@/server/trpc/router";
import type { RuntimeMode } from "@/server/protocol/schema";
import { Toaster } from "@/components/ui/sonner";

export const trpc = createTRPCReact<AppRouter>();

// The runtime travels as a header only when the address names one; otherwise
// the server's configured default applies, which on a deployed control plane
// is the fleet it fronts.
export function getRuntimeHeader(): RuntimeMode | null {
  if (typeof window === "undefined") {
    return null;
  }
  const params = new URLSearchParams(window.location.search);
  const runtime = params.get("runtime");
  if (runtime === "kubernetes" || runtime === "snowflake" || runtime === "local") {
    return runtime;
  }
  return null;
}

/** Answers that asking again cannot change: the request itself was refused. */
const REFUSALS = new Set(["BAD_REQUEST", "UNAUTHORIZED", "FORBIDDEN", "NOT_FOUND"]);

/**
 * One more attempt for a failure that may pass, none for a refusal. A retry
 * waits for the window to have focus, so retrying a refusal would leave a
 * page in a background tab pending instead of saying why it was refused.
 */
export function shouldRetry(failureCount: number, error: unknown): boolean {
  const code = (error as { data?: { code?: unknown } } | null)?.data?.code;
  return failureCount < 1 && !(typeof code === "string" && REFUSALS.has(code));
}

export function AppProviders({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        // Every failed action says so. A mutation that handles its own
        // failure keeps its wording; one that does not is never silent.
        mutationCache: new MutationCache({
          onError: (error, _variables, _context, mutation) => {
            if (!mutation.options.onError) {
              toast.error(error instanceof Error ? error.message : "The action failed");
            }
          },
        }),
        defaultOptions: {
          queries: { refetchOnWindowFocus: false, retry: shouldRetry },
        },
      }),
  );
  const [client] = useState(() =>
    trpc.createClient({
      links: [
        httpBatchLink({
          url: "/api/trpc",
          transformer: superjson,
          headers() {
            const runtime = getRuntimeHeader();
            return runtime ? { "x-flakegraph-runtime": runtime } : {};
          },
          // While the control plane restarts, the edge answers in prose
          // ("no available server"); the parse error that follows says
          // nothing to anyone. Name the outage instead.
          fetch: async (input, init) => {
            const response = await fetch(input, init);
            const type = response.headers.get("content-type") ?? "";
            if (!response.ok && !type.includes("json")) {
              const text = (await response.text()).trim().slice(0, 120);
              throw new Error(
                response.status >= 500
                  ? `The control plane is not answering (${response.status}${text ? `: ${text}` : ""}). It may be restarting; try again in a moment.`
                  : `The control plane refused the request (${response.status}${text ? `: ${text}` : ""}).`,
              );
            }
            return response;
          },
        }),
      ],
    }),
  );
  return (
    <trpc.Provider client={client} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>
        <NuqsAdapter>
          {children}
          <Toaster />
        </NuqsAdapter>
      </QueryClientProvider>
    </trpc.Provider>
  );
}
