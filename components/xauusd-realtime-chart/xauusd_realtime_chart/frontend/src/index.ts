import type { FrontendRenderer, FrontendRendererArgs } from "@streamlit/component-v2-lib";
import Plotly from "plotly.js-dist-min";
import "./style.css";

type AxisViewport = { autorange: boolean; range?: [unknown, unknown] };
type Viewport = Record<string, AxisViewport>;
type PlotElement = HTMLDivElement & {
  on: (event: string, handler: (update: Record<string, unknown>) => void) => void;
  removeListener: (event: string, handler: (update: Record<string, unknown>) => void) => void;
};

export type FrontendState = Record<string, never>;
export type ComponentData = {
  figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> };
  config: Partial<Plotly.Config>;
  viewportRevision: string;
};

type Instance = {
  plot: PlotElement;
  viewport: Viewport;
  revision: string;
  rendering: boolean;
  listenerAttached: boolean;
  pendingData?: ComponentData;
  resizeObserver?: ResizeObserver;
  relayoutHandler: (update: Record<string, unknown>) => void;
};

const instances = new WeakMap<FrontendRendererArgs["parentElement"], Instance>();
const viewportAxes = ["xaxis", "yaxis"];

function axisRange(update: Record<string, unknown>, axis: string): [unknown, unknown] | undefined {
  const combined = update[`${axis}.range`];
  if (Array.isArray(combined) && combined.length === 2) {
    return [combined[0], combined[1]];
  }
  const start = update[`${axis}.range[0]`];
  const end = update[`${axis}.range[1]`];
  return start !== undefined && end !== undefined ? [start, end] : undefined;
}

function captureViewport(instance: Instance, update: Record<string, unknown>): void {
  if (instance.rendering) return;
  for (const axis of viewportAxes) {
    if (update[`${axis}.autorange`] === true) {
      instance.viewport[axis] = { autorange: true };
      continue;
    }
    const range = axisRange(update, axis);
    if (range) {
      instance.viewport[axis] = { autorange: false, range };
    }
  }
}

function applyViewport(layout: Record<string, unknown>, viewport: Viewport): void {
  for (const axis of viewportAxes) {
    const saved = viewport[axis];
    if (!saved) continue;
    const linkedAxes = axis === "xaxis" ? ["xaxis", "xaxis2"] : [axis];
    for (const linkedAxis of linkedAxes) {
      const axisLayout = { ...((layout[linkedAxis] as Record<string, unknown>) || {}) };
      if (saved.autorange) {
        axisLayout.autorange = true;
        delete axisLayout.range;
      } else if (saved.range) {
        axisLayout.autorange = false;
        axisLayout.range = saved.range;
      }
      layout[linkedAxis] = axisLayout;
    }
  }
}

function renderLatest(instance: Instance): void {
  if (instance.rendering || !instance.pendingData) return;
  const data = instance.pendingData;
  instance.pendingData = undefined;
  const layout = structuredClone(data.figure.layout || {}) as Record<string, unknown>;
  applyViewport(layout, instance.viewport);
  layout.autosize = true;
  delete layout.width;
  instance.rendering = true;
  void Plotly.react(instance.plot, data.figure.data || [], layout, data.config || {}).finally(() => {
    instance.rendering = false;
    if (!instance.listenerAttached) {
      instance.plot.on("plotly_relayout", instance.relayoutHandler);
      instance.listenerAttached = true;
    }
    renderLatest(instance);
  });
}

const RealtimeChart: FrontendRenderer<FrontendState, ComponentData> = (args) => {
  const { parentElement, data } = args;
  const plot = parentElement.querySelector<PlotElement>(".chart");
  if (!plot) throw new Error("Realtime chart root was not found");

  let instance = instances.get(parentElement);
  if (!instance) {
    const created: Instance = {
      plot,
      viewport: {},
      revision: data.viewportRevision,
      rendering: false,
      listenerAttached: false,
      relayoutHandler: (_update: Record<string, unknown>): void => {},
    };
    created.relayoutHandler = (update) => captureViewport(created, update);
    created.resizeObserver = new ResizeObserver(() => {
      if (!created.rendering) void Plotly.Plots.resize(created.plot);
    });
    created.resizeObserver.observe(plot);
    instance = created;
    instances.set(parentElement, instance);
  }

  if (instance.revision !== data.viewportRevision) {
    instance.viewport = {};
    instance.revision = data.viewportRevision;
  }

  // Realtime fragment reruns can arrive before Plotly finishes drawing. Keep
  // only the newest payload and render sequentially so stale async draws cannot
  // overwrite the current axes or subplot domains.
  instance.pendingData = data;
  renderLatest(instance);

  return () => {
    // Do not purge on data updates: this persistent node preserves zoom.
  };
};

export default RealtimeChart;
