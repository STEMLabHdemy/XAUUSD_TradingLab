# xauusd-realtime-chart

Local Streamlit Components v2 chart used by XAUUSD TradingLab. It bundles
Plotly.js and updates an existing browser node with `Plotly.react`, preserving
zoom and pan across realtime Streamlit fragment reruns.

The root `requirements.txt` installs this package in editable mode. To rebuild
the committed frontend assets after changing TypeScript:

```powershell
cd .\components\xauusd-realtime-chart\xauusd_realtime_chart\frontend
npm install
npm run build
cd ..\..\..
python -m pip install -e .\components\xauusd-realtime-chart --no-deps --force-reinstall
```

## Build a wheel

To package this component for distribution:

1. Build the frontend assets (from `xauusd_realtime_chart/frontend`):

   ```sh
   npm i
   npm run build
   ```

2. Build the Python wheel using UV (from the project root):
   ```sh
   uv build
   ```

This will create a `dist/` directory containing your wheel. The wheel includes the compiled frontend from `xauusd_realtime_chart/frontend/build`.

### Requirements

- Python >= 3.10
- Node.js >= 24 (LTS)

### Expected output

- `dist/xauusd_realtime_chart-0.0.1-py3-none-any.whl`
- If you run `uv run --with build python -m build` (without `--wheel`), you’ll also get an sdist: `dist/xauusd-realtime-chart-0.0.1.tar.gz`
