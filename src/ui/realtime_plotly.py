from __future__ import annotations

import json

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


_HTML = """
<div class="chart-shell">
  <div class="chart"></div>
  <div class="chart-error" hidden></div>
</div>
"""

_CSS = """
:host { display: block; width: 100%; }
.chart-shell { position: relative; width: 100%; height: 680px; }
.chart { width: 100%; height: 100%; }
.chart-error {
  position: absolute; inset: 0; display: grid; place-items: center;
  color: var(--st-error-text-color, #fca5a5);
  background: var(--st-background-color, #0f172a);
  font-family: var(--st-font, sans-serif);
}
"""

_JS = """
const instances = new WeakMap()
let plotlyLoader = null

function loadPlotly() {
  if (globalThis.Plotly) return Promise.resolve(globalThis.Plotly)
  if (plotlyLoader) return plotlyLoader
  plotlyLoader = new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-xauusd-plotly="3.4.0"]')
    if (existing) {
      existing.addEventListener('load', () => resolve(globalThis.Plotly), { once: true })
      existing.addEventListener('error', reject, { once: true })
      return
    }
    const script = document.createElement('script')
    script.src = 'https://cdn.plot.ly/plotly-3.4.0.min.js'
    script.async = true
    script.dataset.xauusdPlotly = '3.4.0'
    script.onload = () => resolve(globalThis.Plotly)
    script.onerror = () => reject(new Error('Impossibile caricare Plotly.js'))
    document.head.appendChild(script)
  })
  return plotlyLoader
}

function applyViewport(layout, viewport) {
  if (!viewport) return
  for (const axisName of ['xaxis', 'xaxis2', 'yaxis', 'yaxis2']) {
    const saved = viewport[axisName]
    if (!saved) continue
    layout[axisName] = layout[axisName] || {}
    if (saved.autorange) {
      layout[axisName].autorange = true
      delete layout[axisName].range
    } else if (saved.range) {
      layout[axisName].autorange = false
      layout[axisName].range = saved.range
    }
  }
}

function captureViewport(state, update) {
  if (state.rendering) return
  const mapping = {
    xaxis: 'xaxis', xaxis2: 'xaxis2', yaxis: 'yaxis', yaxis2: 'yaxis2',
  }
  for (const [prefix, axisName] of Object.entries(mapping)) {
    if (update[`${prefix}.autorange`] === true) {
      state.viewport[axisName] = { autorange: true }
      continue
    }
    const start = update[`${prefix}.range[0]`]
    const end = update[`${prefix}.range[1]`]
    if (start !== undefined && end !== undefined) {
      state.viewport[axisName] = { autorange: false, range: [start, end] }
    }
  }
}

export default function(component) {
  const { data, parentElement } = component
  const plot = parentElement.querySelector('.chart')
  const error = parentElement.querySelector('.chart-error')
  if (!plot || !data?.figure) return

  let state = instances.get(parentElement)
  if (!state) {
    state = { viewport: {}, revision: null, rendering: false, handler: null }
    instances.set(parentElement, state)
  }
  if (state.revision !== data.viewport_revision) {
    state.viewport = {}
    state.revision = data.viewport_revision
  }

  let active = true
  loadPlotly().then(async (Plotly) => {
    if (!active) return
    error.hidden = true
    const layout = structuredClone(data.figure.layout || {})
    applyViewport(layout, state.viewport)
    state.rendering = true
    await Plotly.react(plot, data.figure.data || [], layout, data.config || {})
    state.rendering = false
    if (!state.handler) {
      state.handler = (update) => captureViewport(state, update)
      plot.on('plotly_relayout', state.handler)
    }
  }).catch((reason) => {
    if (!active) return
    error.hidden = false
    error.textContent = `Grafico realtime non disponibile: ${reason.message}`
  })

  return () => { active = false }
}
"""

_REALTIME_PLOTLY = st.components.v2.component(
    "xauusd_realtime_plotly",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def realtime_plotly_chart(
    figure: go.Figure,
    *,
    key: str,
    viewport_revision: str,
    config: dict[str, object] | None = None,
) -> None:
    """Update one browser-resident Plotly graph without remounting its DOM."""
    payload = json.loads(pio.to_json(figure, validate=False, remove_uids=True))
    payload.get("layout", {}).pop("template", None)
    _REALTIME_PLOTLY(
        key=key,
        data={
            "figure": payload,
            "viewport_revision": viewport_revision,
            "config": config or {},
        },
        width="stretch",
        height=680,
    )
