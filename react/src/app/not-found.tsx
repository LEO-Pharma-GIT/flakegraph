import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-3 p-8">
      <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Control plane</p>
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="text-sm text-muted-foreground">That control-plane route does not exist.</p>
      <Button asChild>
        <Link href="/">Back to graphs</Link>
      </Button>
    </main>
  );
}
