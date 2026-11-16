"use client";

import { Button } from "@/components/ui/button";
import { nextPageCount } from "@/lib/paging";

/**
 * The footer under a paged list: how much of it is on screen, and one
 * click for the next page. Renders nothing once everything is shown, so a
 * short list has no footer.
 */
export function ShowMore({
  shown,
  total,
  pageSize,
  noun,
  onMore,
}: {
  shown: number;
  total: number;
  pageSize: number;
  /** Plural, as in "Showing 10 of 34 versions". */
  noun: string;
  onMore: () => void;
}) {
  if (shown >= total) {
    return null;
  }
  return (
    <div className="flex items-center gap-3 text-xs text-muted-foreground" data-testid="show-more">
      <span>
        Showing {shown.toLocaleString()} of {total.toLocaleString()} {noun}
      </span>
      <Button size="sm" variant="outline" className="h-7" onClick={onMore}>
        Show {nextPageCount(shown, total, pageSize)} more
      </Button>
    </div>
  );
}
