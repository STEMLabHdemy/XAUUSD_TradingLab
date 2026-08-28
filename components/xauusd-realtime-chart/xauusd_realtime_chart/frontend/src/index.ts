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
  relayoutHandler: (update: Record<string, unknown>) => void;
};

const instances = new WeakMap<FrontendRendererArgs["parentElement"], Instance>();
const axes = ["xaxis", "xaxis2", "yaxis", "yaxis2"];

function captureViewport(instance: Instance, update: Record<string, unknown>): void {
  if (instance.rendering) return;
  for (const axis of axes) {
    if (update[`${axis}.autorange`] === true) {
      instance.viewport[axis] = { autorange: true };
      continue;
    }
    const start = update[`${axis}.range[0]`];
    const end = update[`${axis}.range[1]`];
    if (start !== undefined && end !== undefined) {
      instance.viewport[axis] = { autorange: false, range: [start, end] };
    }
  }
}

function applyViewport(layout: Record<string, unknown>, viewport: Viewport): void {
  for (const axis of axes) {
    const saved = viewport[axis];
    if (!saved) continue;
    const axisLayout = { ...((layout[axis] as Record<string, unknown>) || {}) };
    if (saved.autorange) {
      axisLayout.autorange = true;
      delete axisLayout.range;
    } else if (saved.range) {
      axisLayout.autorange = false;
      axisLayout.range = saved.range;
    }
    layout[axis] = axisLayout;
  }
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
    instance = created;
    instances.set(parentElement, instance);
  }

  if (instance.revision !== data.viewportRevision) {
    instance.viewport = {};
    instance.revision = data.viewportRevision;
  }

  const layout = structuredClone(data.figure.layout || {}) as Record<string, unknown>;
  applyViewport(layout, instance.viewport);
  instance.rendering = true;
  void Plotly.react(plot, data.figure.data || [], layout, data.config || {}).finally(() => {
    if (instance) {
      instance.rendering = false;
      if (!instance.listenerAttached) {
        plot.on("plotly_relayout", instance.relayoutHandler);
        instance.listenerAttached = true;
      }
    }
  });

  return () => {
    // Do not purge on data updates: this persistent node preserves zoom.
  };
};

export default RealtimeChart;
