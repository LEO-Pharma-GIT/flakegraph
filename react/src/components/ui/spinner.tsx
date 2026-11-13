import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * The one sign that something is under way. Decorative by default: the
 * control it sits in says what is happening ("Deleting…"), and a status that
 * stands alone passes `label` so a screen reader hears it too.
 */
export function Spinner({ className, label }: { className?: string; label?: string }) {
  return (
    <Loader2
      className={cn("size-3.5 shrink-0 animate-spin", className)}
      role={label ? "status" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    />
  );
}
