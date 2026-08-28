from __future__ import annotations

from typing import Any

import streamlit as st


_component = st.components.v2.component(
    "xauusd-realtime-chart.xauusd_realtime_chart",
    js="index-*.js",
    css="index-*.css",
    html="""
        <div class="component-root">
            <div class="chart"></div>
        </div>
    """,
)


def xauusd_realtime_chart(
    figure: dict[str, Any],
    *,
    viewport_revision: str,
    config: dict[str, Any] | None = None,
    key: str | None = None,
) -> None:
    """Render a browser-resident Plotly chart updated without DOM remounts."""
    _component(
        key=key,
        data={
            "figure": figure,
            "viewportRevision": viewport_revision,
            "config": config or {},
        },
        width="stretch",
        height=680,
    )
