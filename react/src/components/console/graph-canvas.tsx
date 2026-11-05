"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Map as MapIcon, Maximize2, Minus, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  distanceToSegment,
  easeInOut,
  fitCamera,
  layoutGraph,
  nHopNodeIds,
  nodeColor,
  normalizeGraph,
  selectionSeeds,
  worldToScreen,
  zoomAt,
  type ColorMode,
  type FocusDepth,
  type GraphCamera,
  type GraphHover,
  type LayoutEdge,
  type LayoutMode,
  type LayoutNode,
  type PathHighlight,
} from "@/lib/graph-geometry";

interface GraphCanvasProps {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  compact?: boolean;
  colorMode?: ColorMode;
  layoutMode?: LayoutMode;
  focusDepth?: FocusDepth;
  pathHighlight?: PathHighlight | null;
  hiddenTypes?: Set<string>;
  hiddenRelationTypes?: Set<string>;
  membership?: Map<string, Set<string>>;
  flashNonce?: number;
  showMinimap?: boolean;
  onToggleMinimap?: () => void;
  onHover?: (payload: GraphHover | null) => void;
  onShiftClickNode?: (id: string) => void;
}

interface Hit {
  kind: "node" | "edge";
  id: string;
}

const FLY_RATIO = 0.55;
const FLY_MS = 420;

export function GraphCanvas({
  nodes,
  edges,
  selectedId,
  onSelect,
  compact = false,
  colorMode = "type",
  layoutMode = "force",
  focusDepth = null,
  pathHighlight = null,
  hiddenTypes,
  hiddenRelationTypes,
  membership,
  flashNonce = 0,
  showMinimap = false,
  onToggleMinimap,
  onHover,
  onShiftClickNode,
}: GraphCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const minimapRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<GraphCamera>({ x: 0, y: 0, scale: 1 });
  const animationRef = useRef<number | null>(null);
  const dragRef = useRef<{ pointerId: number; lastX: number; lastY: number; moved: boolean; hit: Hit | null } | null>(null);
  const hoverIdRef = useRef<string | null>(null);
  const fittedSignatureRef = useRef("");
  const [size, setSize] = useState({ width: 900, height: compact ? 280 : 520 });
  const [camera, setCamera] = useState<GraphCamera>({ x: 0, y: 0, scale: 1 });
  const [hoverId, setHoverId] = useState<string | null>(null);

  const graph = useMemo(() => {
    const normalized = normalizeGraph(nodes, edges, membership ?? new Map());
    return {
      nodes: layoutGraph(normalized.nodes, normalized.edges, layoutMode),
      edges: normalized.edges,
    };
  }, [edges, layoutMode, membership, nodes]);

  const byId = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph.nodes]);
  const maxDegree = useMemo(() => Math.max(1, ...graph.nodes.map((node) => node.degree)), [graph.nodes]);
  const visibleNodes = useMemo(
    () => graph.nodes.filter((node) => !hiddenTypes?.has(node.type)),
    [graph.nodes, hiddenTypes],
  );
  const visibleNodeIds = useMemo(() => new Set(visibleNodes.map((node) => node.id)), [visibleNodes]);
  const visibleEdges = useMemo(
    () =>
      graph.edges.filter((edge) => {
        if (hiddenRelationTypes?.has(edge.relationType)) {
          return false;
        }
        return visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target);
      }),
    [graph.edges, hiddenRelationTypes, visibleNodeIds],
  );

  const focusSet = useMemo(() => {
    if (!focusDepth || !selectedId) {
      return null;
    }
    return nHopNodeIds(visibleEdges, selectionSeeds(selectedId, visibleEdges), focusDepth);
  }, [focusDepth, selectedId, visibleEdges]);

  const applyCamera = useCallback((next: GraphCamera) => {
    cameraRef.current = next;
    setCamera(next);
  }, []);

  const stopAnimation = useCallback(() => {
    if (animationRef.current != null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, []);

  const animateCamera = useCallback(
    (target: GraphCamera, duration = FLY_MS) => {
      stopAnimation();
      const from = cameraRef.current;
      const started = performance.now();
      const tick = (now: number) => {
        const t = Math.min(1, (now - started) / duration);
        const eased = easeInOut(t);
        applyCamera({
          x: from.x + (target.x - from.x) * eased,
          y: from.y + (target.y - from.y) * eased,
          scale: from.scale + (target.scale - from.scale) * eased,
        });
        if (t < 1) {
          animationRef.current = requestAnimationFrame(tick);
        } else {
          animationRef.current = null;
        }
      };
      animationRef.current = requestAnimationFrame(tick);
    },
    [applyCamera, stopAnimation],
  );

  const fitView = useCallback(
    (animate = false) => {
      const next = fitCamera(visibleNodes, size.width, size.height);
      if (animate) {
        animateCamera(next);
      } else {
        stopAnimation();
        applyCamera(next);
      }
    },
    [animateCamera, applyCamera, size.height, size.width, stopAnimation, visibleNodes],
  );

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) {
      return;
    }
    const update = () => {
      const width = Math.max(320, Math.round(wrap.clientWidth));
      const height = compact
        ? Math.max(220, Math.min(320, Math.round(width * 0.28)))
        : Math.max(420, Math.min(640, Math.round(width * 0.48)));
      setSize((current) => (current.width === width && current.height === height ? current : { width, height }));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [compact]);

  useEffect(() => {
    const signature = `${layoutMode}:${visibleNodes.map((node) => node.id).join(",")}`;
    if (!visibleNodes.length) {
      fittedSignatureRef.current = signature;
      return;
    }
    if (fittedSignatureRef.current === signature) {
      return;
    }
    fittedSignatureRef.current = signature;
    applyCamera(fitCamera(visibleNodes, size.width, size.height));
  }, [applyCamera, layoutMode, size.height, size.width, visibleNodes]);

  useEffect(() => {
    if (!selectedId || flashNonce <= 0) {
      return;
    }
    const node = byId.get(selectedId);
    if (!node) {
      return;
    }
    animateCamera({ x: node.x, y: node.y, scale: Math.max(cameraRef.current.scale, FLY_RATIO) });
  }, [animateCamera, byId, flashNonce, selectedId]);

  useEffect(() => {
    return () => stopAnimation();
  }, [stopAnimation]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const { width, height } = size;
    const onNativeWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * width;
      const y = ((event.clientY - rect.top) / rect.height) * height;
      stopAnimation();
      applyCamera(zoomAt(cameraRef.current, x, y, width, height, event.deltaY < 0 ? 1.12 : 1 / 1.12));
    };
    canvas.addEventListener("wheel", onNativeWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onNativeWheel);
  }, [applyCamera, size, stopAnimation]);

  const hitTest = useCallback(
    (clientX: number, clientY: number): Hit | null => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return null;
      }
      const rect = canvas.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * size.width;
      const y = ((clientY - rect.top) / rect.height) * size.height;
      const current = cameraRef.current;
      let nearest: LayoutNode | null = null;
      let best = 16;
      for (const node of visibleNodes) {
        const point = worldToScreen(current, node.x, node.y, size.width, size.height);
        const distance = Math.hypot(point.x - x, point.y - y);
        const threshold = Math.max(12, node.size + 4);
        if (distance < best && distance <= threshold) {
          best = distance;
          nearest = node;
        }
      }
      if (nearest) {
        return { kind: "node", id: nearest.id };
      }
      let edgeHit: LayoutEdge | null = null;
      let edgeBest = 7;
      for (const edge of visibleEdges) {
        const source = byId.get(edge.source);
        const target = byId.get(edge.target);
        if (!source || !target) {
          continue;
        }
        const from = worldToScreen(current, source.x, source.y, size.width, size.height);
        const to = worldToScreen(current, target.x, target.y, size.width, size.height);
        const distance = distanceToSegment(x, y, from.x, from.y, to.x, to.y);
        if (distance < edgeBest) {
          edgeBest = distance;
          edgeHit = edge;
        }
      }
      return edgeHit ? { kind: "edge", id: edgeHit.id } : null;
    },
    [byId, size.height, size.width, visibleEdges, visibleNodes],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintGraph(context, {
      width: size.width,
      height: size.height,
      camera,
      nodes: visibleNodes,
      edges: visibleEdges,
      byId,
      selectedId,
      hoverId,
      colorMode,
      maxDegree,
      focusSet,
      pathHighlight,
    });
    if (showMinimap) {
      paintMinimap(minimapRef.current, {
        camera,
        nodes: visibleNodes,
        width: size.width,
        height: size.height,
        colorMode,
        maxDegree,
      });
    }
  }, [
    byId,
    camera,
    colorMode,
    focusSet,
    hoverId,
    maxDegree,
    pathHighlight,
    selectedId,
    showMinimap,
    size.height,
    size.width,
    visibleEdges,
    visibleNodes,
  ]);

  function zoom(factor: number, origin?: { x: number; y: number }) {
    stopAnimation();
    const point = origin ?? { x: size.width / 2, y: size.height / 2 };
    applyCamera(zoomAt(cameraRef.current, point.x, point.y, size.width, size.height, factor));
  }

  function onPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0) {
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    canvas.setPointerCapture(event.pointerId);
    stopAnimation();
    dragRef.current = {
      pointerId: event.pointerId,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
      hit: hitTest(event.clientX, event.clientY),
    };
  }

  function onPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId) {
      const dx = event.clientX - drag.lastX;
      const dy = event.clientY - drag.lastY;
      if (Math.hypot(dx, dy) > 3) {
        drag.moved = true;
      }
      if (drag.moved) {
        const current = cameraRef.current;
        applyCamera({
          ...current,
          x: current.x - dx / current.scale,
          y: current.y - dy / current.scale,
        });
        drag.lastX = event.clientX;
        drag.lastY = event.clientY;
        onHover?.(null);
      }
      return;
    }
    const hit = hitTest(event.clientX, event.clientY);
    const nextId = hit?.id ?? null;
    if (hoverIdRef.current !== nextId) {
      hoverIdRef.current = nextId;
      setHoverId(nextId);
    }
    onHover?.(
      hit
        ? { kind: hit.kind, id: hit.id, x: event.clientX, y: event.clientY }
        : null,
    );
  }

  function onPointerUp(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    if (drag.moved) {
      return;
    }
    const hit = drag.hit;
    if (!hit) {
      onSelect(null);
      return;
    }
    if (hit.kind === "node" && event.shiftKey) {
      onShiftClickNode?.(hit.id);
      return;
    }
    onSelect(hit.id);
  }

  function onPointerLeave() {
    if (dragRef.current) {
      return;
    }
    hoverIdRef.current = null;
    setHoverId(null);
    onHover?.(null);
  }

  function onDoubleClick(event: ReactMouseEvent<HTMLCanvasElement>) {
    const hit = hitTest(event.clientX, event.clientY);
    if (hit?.kind === "node") {
      const node = byId.get(hit.id);
      if (node) {
        onSelect(hit.id);
        animateCamera({ x: node.x, y: node.y, scale: Math.min(2.4, Math.max(1.1, cameraRef.current.scale * 1.4)) });
      }
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    zoom(1.35, {
      x: ((event.clientX - rect.left) / rect.width) * size.width,
      y: ((event.clientY - rect.top) / rect.height) * size.height,
    });
  }

  function onKeyDown(event: KeyboardEvent<HTMLCanvasElement>) {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoom(1.18);
      return;
    }
    if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      zoom(1 / 1.18);
      return;
    }
    if (event.key === "0") {
      event.preventDefault();
      fitView(true);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onSelect(null);
      return;
    }
    const panKeys: Record<string, { x: number; y: number }> = {
      ArrowLeft: { x: -48, y: 0 },
      ArrowRight: { x: 48, y: 0 },
      ArrowUp: { x: 0, y: -48 },
      ArrowDown: { x: 0, y: 48 },
    };
    const pan = panKeys[event.key];
    if (pan) {
      event.preventDefault();
      const current = cameraRef.current;
      applyCamera({
        ...current,
        x: current.x + pan.x / current.scale,
        y: current.y + pan.y / current.scale,
      });
      return;
    }
    if ((event.key === "[" || event.key === "]") && visibleNodes.length) {
      event.preventDefault();
      const index = visibleNodes.findIndex((node) => node.id === selectedId);
      const delta = event.key === "]" ? 1 : -1;
      const next = visibleNodes[(index + delta + visibleNodes.length) % visibleNodes.length];
      if (next) {
        onSelect(next.id);
        animateCamera({ x: next.x, y: next.y, scale: Math.max(cameraRef.current.scale, FLY_RATIO) });
      }
    }
  }

  function onMinimapPointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    const canvas = minimapRef.current;
    if (!canvas || !visibleNodes.length) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 160;
    const y = ((event.clientY - rect.top) / rect.height) * 112;
    const bounds = nodeBounds(visibleNodes);
    const worldX = bounds.minX + (x / 160) * (bounds.maxX - bounds.minX || 1);
    const worldY = bounds.minY + (y / 112) * (bounds.maxY - bounds.minY || 1);
    applyCamera({ ...cameraRef.current, x: worldX, y: worldY });
  }

  const empty = visibleNodes.length === 0;

  return (
    <div ref={wrapRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        className="w-full cursor-grab rounded-lg border border-border bg-card active:cursor-grabbing"
        style={{ height: size.height, touchAction: "none" }}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerLeave}
        onPointerLeave={onPointerLeave}
        onDoubleClick={onDoubleClick}
        onKeyDown={onKeyDown}
        data-testid="graph-canvas"
        data-zoom={camera.scale.toFixed(2)}
        aria-label={`Graph canvas, ${visibleNodes.length} entities. Scroll to zoom, drag to pan. Plus and minus zoom, 0 fits the view, arrows pan, brackets move the selection. Shift-click two entities to find a path.`}
      />
      {!empty ? (
        <div className="absolute top-2 right-2 z-10 flex flex-col gap-1 rounded-md border border-border bg-background/95 p-1 shadow-sm backdrop-blur-sm">
          <Button size="icon" variant="ghost" className="size-8" aria-label="Zoom in" data-testid="graph-zoom-in" onClick={() => zoom(1.2)}>
            <Plus />
          </Button>
          <Button size="icon" variant="ghost" className="size-8" aria-label="Zoom out" data-testid="graph-zoom-out" onClick={() => zoom(1 / 1.2)}>
            <Minus />
          </Button>
          <Button size="icon" variant="ghost" className="size-8" aria-label="Fit graph" data-testid="graph-fit" onClick={() => fitView(true)}>
            <Maximize2 />
          </Button>
          {onToggleMinimap ? (
            <Button
              size="icon"
              variant={showMinimap ? "secondary" : "ghost"}
              className="size-8"
              aria-label={showMinimap ? "Hide minimap" : "Show minimap"}
              aria-pressed={showMinimap}
              data-testid="graph-minimap-toggle"
              onClick={onToggleMinimap}
            >
              <MapIcon />
            </Button>
          ) : null}
        </div>
      ) : null}
      {showMinimap && !empty ? (
        <canvas
          ref={minimapRef}
          width={160}
          height={112}
          className="absolute right-2 bottom-2 z-10 cursor-pointer rounded-md border border-border bg-background/90 shadow-sm"
          data-testid="graph-minimap"
          aria-label="Graph minimap"
          onPointerDown={onMinimapPointer}
        />
      ) : null}
    </div>
  );
}

function nodeBounds(nodes: LayoutNode[]) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x);
    maxY = Math.max(maxY, node.y);
  }
  if (!Number.isFinite(minX)) {
    return { minX: -1, minY: -1, maxX: 1, maxY: 1 };
  }
  return { minX, minY, maxX, maxY };
}

function paintGraph(
  context: CanvasRenderingContext2D,
  input: {
    width: number;
    height: number;
    camera: GraphCamera;
    nodes: LayoutNode[];
    edges: LayoutEdge[];
    byId: Map<string, LayoutNode>;
    selectedId: string | null;
    hoverId: string | null;
    colorMode: ColorMode;
    maxDegree: number;
    focusSet: Set<string> | null;
    pathHighlight: PathHighlight | null;
  },
) {
  const styles = getComputedStyle(context.canvas);
  const card = styles.getPropertyValue("--card").trim() || "#fff";
  const foreground = styles.getPropertyValue("--foreground").trim() || "#17212b";
  const muted = styles.getPropertyValue("--muted-foreground").trim() || "#94a3b8";
  const border = styles.getPropertyValue("--border").trim() || "#dfe5e8";
  context.clearRect(0, 0, input.width, input.height);
  context.fillStyle = card;
  context.fillRect(0, 0, input.width, input.height);

  const labelBudget = input.nodes.length <= 50 || input.camera.scale >= 1.15 ? 80 : input.camera.scale >= 0.7 ? 24 : 8;

  for (const edge of input.edges) {
    const source = input.byId.get(edge.source);
    const target = input.byId.get(edge.target);
    if (!source || !target) {
      continue;
    }
    const from = worldToScreen(input.camera, source.x, source.y, input.width, input.height);
    const to = worldToScreen(input.camera, target.x, target.y, input.width, input.height);
    const isPath = Boolean(input.pathHighlight?.edgeIds.has(edge.id));
    const inFocus = !input.focusSet || (input.focusSet.has(edge.source) && input.focusSet.has(edge.target));
    const incident =
      !input.selectedId ||
      edge.id === input.selectedId ||
      edge.source === input.selectedId ||
      edge.target === input.selectedId;
    if (input.focusSet && !inFocus && !isPath) {
      continue;
    }
    const faded = Boolean(input.pathHighlight) ? !isPath : Boolean(input.selectedId) && !incident;
    context.strokeStyle = isPath ? "#d39a25" : faded ? border : muted;
    context.fillStyle = context.strokeStyle;
    context.lineWidth = isPath ? 2.4 : faded ? 1 : 1.4;
    context.globalAlpha = faded ? 0.35 : 1;
    drawArrow(
      context,
      from.x,
      from.y,
      to.x,
      to.y,
      source.size * (input.camera.scale / 2 + 0.5),
      target.size * (input.camera.scale / 2 + 0.5),
      edge.claimed,
    );
    if (!faded && input.camera.scale >= 1.05 && input.edges.length <= 80) {
      context.fillStyle = muted;
      context.font = "10px ui-sans-serif, system-ui, sans-serif";
      context.fillText(edge.relationType.slice(0, 22), (from.x + to.x) / 2 + 4, (from.y + to.y) / 2 - 4);
    }
    context.globalAlpha = 1;
  }

  const labeled = new Set<string>();
  if (input.selectedId) labeled.add(input.selectedId);
  if (input.hoverId) labeled.add(input.hoverId);

  for (const node of input.nodes) {
    const point = worldToScreen(input.camera, node.x, node.y, input.width, input.height);
    if (point.x < -24 || point.y < -24 || point.x > input.width + 24 || point.y > input.height + 24) {
      continue;
    }
    const selected = node.id === input.selectedId;
    const hovered = node.id === input.hoverId;
    const isPath = Boolean(input.pathHighlight?.nodeIds.has(node.id));
    const inFocus = !input.focusSet || input.focusSet.has(node.id);
    const dimmed = (!inFocus && !isPath) || (Boolean(input.pathHighlight) && !isPath && !selected);
    const radius = (selected || hovered ? node.size + 1.5 : node.size) * Math.min(1.35, 0.55 + input.camera.scale * 0.45);
    context.globalAlpha = dimmed ? 0.28 : 1;
    context.beginPath();
    context.fillStyle = isPath ? "#d39a25" : selected ? "#16806a" : nodeColor(node, input.colorMode, input.maxDegree);
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
    if (selected || hovered) {
      context.lineWidth = 2;
      context.strokeStyle = foreground;
      context.stroke();
    }
    const showLabel = selected || hovered || isPath || labeled.size < labelBudget;
    if (showLabel) {
      labeled.add(node.id);
      context.fillStyle = dimmed ? muted : foreground;
      context.font = `${selected ? 12 : 11}px ui-sans-serif, system-ui, sans-serif`;
      context.fillText(node.label.slice(0, 32), Math.min(point.x + radius + 5, input.width - 86), point.y + 4);
    }
    context.globalAlpha = 1;
  }
}

function drawArrow(
  context: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  sourceRadius: number,
  targetRadius: number,
  dashed = false,
) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const startX = x1 + Math.cos(angle) * sourceRadius;
  const startY = y1 + Math.sin(angle) * sourceRadius;
  const endX = x2 - Math.cos(angle) * (targetRadius + 1);
  const endY = y2 - Math.sin(angle) * (targetRadius + 1);
  context.beginPath();
  context.moveTo(startX, startY);
  context.lineTo(endX, endY);
  // A claimed relation is drawn dashed; its arrowhead stays solid.
  context.setLineDash(dashed ? [5, 4] : []);
  context.stroke();
  context.setLineDash([]);
  const head = 6;
  context.beginPath();
  context.moveTo(endX, endY);
  context.lineTo(endX - head * Math.cos(angle - 0.4), endY - head * Math.sin(angle - 0.4));
  context.lineTo(endX - head * Math.cos(angle + 0.4), endY - head * Math.sin(angle + 0.4));
  context.closePath();
  context.fill();
}

function paintMinimap(
  canvas: HTMLCanvasElement | null,
  input: {
    camera: GraphCamera;
    nodes: LayoutNode[];
    width: number;
    height: number;
    colorMode: ColorMode;
    maxDegree: number;
  },
) {
  if (!canvas) {
    return;
  }
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  const width = 160;
  const height = 112;
  context.clearRect(0, 0, width, height);
  const styles = getComputedStyle(canvas);
  context.fillStyle = styles.getPropertyValue("--card").trim() || "#fff";
  context.fillRect(0, 0, width, height);
  const bounds = nodeBounds(input.nodes);
  const spanX = bounds.maxX - bounds.minX || 1;
  const spanY = bounds.maxY - bounds.minY || 1;
  for (const node of input.nodes) {
    const x = ((node.x - bounds.minX) / spanX) * width;
    const y = ((node.y - bounds.minY) / spanY) * height;
    context.beginPath();
    context.fillStyle = nodeColor(node, input.colorMode, input.maxDegree);
    context.arc(x, y, 2.2, 0, Math.PI * 2);
    context.fill();
  }
  const worldLeft = input.camera.x - input.width / 2 / input.camera.scale;
  const worldTop = input.camera.y - input.height / 2 / input.camera.scale;
  const worldRight = input.camera.x + input.width / 2 / input.camera.scale;
  const worldBottom = input.camera.y + input.height / 2 / input.camera.scale;
  const vx = ((worldLeft - bounds.minX) / spanX) * width;
  const vy = ((worldTop - bounds.minY) / spanY) * height;
  const vw = ((worldRight - worldLeft) / spanX) * width;
  const vh = ((worldBottom - worldTop) / spanY) * height;
  context.strokeStyle = "#16806a";
  context.lineWidth = 1.5;
  context.strokeRect(vx, vy, vw, vh);
}
