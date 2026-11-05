import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

export function Alert({
  className,
  variant = "default",
  ...props
}: HTMLAttributes<HTMLDivElement> & { variant?: "default" | "destructive" | "success" | "warning" }) {
  const variants = {
    default: "border-border bg-muted/40",
    destructive: "border-destructive/40 bg-destructive/10 text-destructive",
    success: "border-emerald-300 bg-emerald-50 text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-100",
    warning: "border-amber-300 bg-amber-50 text-amber-900 dark:bg-amber-950/40 dark:text-amber-100",
  };
  return (
    <div role="alert" className={cn("rounded-md border px-3 py-2 text-sm", variants[variant], className)} {...props} />
  );
}
