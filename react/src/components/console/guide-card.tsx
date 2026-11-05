"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export interface GuideAction {
  label: string;
  onClick: () => void;
  variant?: "default" | "secondary" | "outline" | "ghost";
  /** The action is under way: shown with a spinner and not pressed twice. */
  pending?: boolean;
}

export function GuideCard({
  title,
  why,
  actions,
  onDismiss,
  tone = "default",
}: {
  title: string;
  why: string;
  actions: GuideAction[];
  onDismiss?: () => void;
  tone?: "default" | "warning";
}) {
  return (
    <Card
      data-testid="guide-card"
      className={
        tone === "warning"
          ? "border-amber-400/60 bg-amber-50/80"
          : "border-border border-l-2 border-l-primary bg-card"
      }
    >
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-primary">Next</p>
          <CardTitle className="text-sm">{title}</CardTitle>
          <CardDescription>{why}</CardDescription>
        </div>
        {onDismiss ? (
          <Button size="sm" variant="ghost" onClick={onDismiss} aria-label="Dismiss suggestion">
            Dismiss
          </Button>
        ) : null}
      </CardHeader>
      {actions.length ? (
        <CardContent className="flex flex-wrap gap-2">
          {actions.map((action) => (
            <Button
              key={action.label}
              size="sm"
              variant={action.variant ?? "default"}
              pending={action.pending}
              onClick={action.onClick}
            >
              {action.label}
            </Button>
          ))}
        </CardContent>
      ) : null}
    </Card>
  );
}
