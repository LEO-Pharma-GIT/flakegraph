// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDebouncedValue } from "./use-debounced-value";

type Source = { kind: string; bucket: string };

function mount() {
  const seen: Source[] = [];
  let render: (value: Source) => void = () => {};
  function Probe({ value }: { value: Source }) {
    seen.push(useDebouncedValue(value, 500));
    return null;
  }
  const container = document.createElement("div");
  const root: Root = createRoot(container);
  render = (value) => act(() => root.render(<Probe value={value} />));
  return {
    render,
    latest: () => seen.at(-1),
    unmount: () => act(() => root.unmount()),
  };
}

describe("useDebouncedValue", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  });
  afterEach(() => vi.useRealTimers());

  it("settles on the last value once typing pauses", () => {
    const probe = mount();
    probe.render({ kind: "s3", bucket: "" });
    expect(probe.latest()).toEqual({ kind: "s3", bucket: "" });
    probe.render({ kind: "s3", bucket: "t" });
    probe.render({ kind: "s3", bucket: "te" });
    act(() => vi.advanceTimersByTime(400));
    expect(probe.latest()).toEqual({ kind: "s3", bucket: "" });
    probe.render({ kind: "s3", bucket: "test" });
    act(() => vi.advanceTimersByTime(400));
    expect(probe.latest()).toEqual({ kind: "s3", bucket: "" });
    act(() => vi.advanceTimersByTime(100));
    expect(probe.latest()).toEqual({ kind: "s3", bucket: "test" });
    probe.unmount();
  });

  it("does not restart the wait for an equal object rebuilt each render", () => {
    const probe = mount();
    probe.render({ kind: "s3", bucket: "a" });
    probe.render({ kind: "s3", bucket: "b" });
    act(() => vi.advanceTimersByTime(300));
    probe.render({ kind: "s3", bucket: "b" });
    act(() => vi.advanceTimersByTime(200));
    expect(probe.latest()).toEqual({ kind: "s3", bucket: "b" });
    probe.unmount();
  });
});
