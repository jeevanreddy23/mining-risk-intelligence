from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .inference import score_packet
from .schemas import SensorPacket


app = FastAPI(title="Edge Mining Risk Pipeline", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = OUTPUTS_DIR / "plots"

if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(sensor_packet: SensorPacket) -> dict[str, object]:
    return score_packet(sensor_packet.model_dump())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    metrics = _load_json(DATA_DIR / "metrics.json")
    white_check = _load_json(OUTPUTS_DIR / "white_reality_check" / "white_reality_check.json")
    candidate_models = metrics.get("candidate_models", {})

    plot_files = [
        ("Pre-training label distribution", "/outputs/plots/label_distribution.png"),
        ("Pre-training correlation heatmap", "/outputs/plots/correlation_heatmap.png"),
        ("Post-training confusion matrix", "/outputs/plots/confusion_matrix.png"),
        ("Post-training prediction confidence", "/outputs/plots/prediction_confidence.png"),
    ]
    plot_cards = "".join(
        f"""
        <div class="plot-card">
          <h3>{title}</h3>
          <img src="{src}" alt="{title}" />
        </div>
        """
        for title, src in plot_files
        if (PLOTS_DIR / Path(src).name).exists()
    )

    model_rows = "".join(
        f"""
        <tr>
          <td>{name}</td>
          <td>{values.get('accuracy', 0):.3f}</td>
          <td>{values.get('weighted_f1', 0):.3f}</td>
          <td>{values.get('macro_f1', 0):.3f}</td>
        </tr>
        """
        for name, values in candidate_models.items()
    )

    selected_model = metrics.get("selected_model", "not trained")
    selected_accuracy = metrics.get("selected_model_accuracy", 0)
    selected_macro_f1 = metrics.get("selected_model_macro_f1", 0)
    white_p = white_check.get("bootstrap_p_value", "n/a")
    white_text = white_check.get("interpretation", "White Reality Check not yet run.")

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Mining Risk Intelligence Dashboard</title>
      <style>
        :root {{
          --bg: #f6f4ef;
          --panel: #ffffff;
          --ink: #192126;
          --muted: #5a6872;
          --line: #d9d4c6;
          --accent: #b85c38;
          --accent-2: #2d6a73;
        }}
        body {{
          margin: 0;
          font-family: Georgia, "Segoe UI", serif;
          background: linear-gradient(180deg, #f1eee6 0%, #fbfaf7 100%);
          color: var(--ink);
        }}
        .wrap {{
          max-width: 1200px;
          margin: 0 auto;
          padding: 32px 24px 56px;
        }}
        .hero {{
          display: grid;
          grid-template-columns: 1.6fr 1fr;
          gap: 24px;
          margin-bottom: 24px;
        }}
        .panel {{
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 20px;
          box-shadow: 0 8px 24px rgba(25, 33, 38, 0.06);
          padding: 24px;
        }}
        h1, h2, h3 {{
          margin-top: 0;
        }}
        .sub {{
          color: var(--muted);
          line-height: 1.5;
        }}
        .metrics {{
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 14px;
          margin-top: 18px;
        }}
        .metric {{
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 14px;
          background: #fcfbf8;
        }}
        .metric .label {{
          color: var(--muted);
          font-size: 0.85rem;
        }}
        .metric .value {{
          font-size: 1.5rem;
          font-weight: 700;
          margin-top: 4px;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          margin-top: 10px;
        }}
        th, td {{
          text-align: left;
          padding: 10px 8px;
          border-bottom: 1px solid var(--line);
          font-size: 0.95rem;
        }}
        .plots {{
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 18px;
          margin-top: 22px;
        }}
        .plot-card {{
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 18px;
          box-shadow: 0 8px 24px rgba(25, 33, 38, 0.05);
        }}
        .plot-card img {{
          width: 100%;
          border-radius: 12px;
          border: 1px solid var(--line);
        }}
        .pill {{
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          background: #efe7d8;
          color: #6f3b1f;
          font-size: 0.85rem;
          margin-right: 8px;
          margin-bottom: 8px;
        }}
        .actions a {{
          display: inline-block;
          margin-right: 12px;
          margin-top: 12px;
          color: white;
          background: var(--accent);
          text-decoration: none;
          padding: 10px 14px;
          border-radius: 12px;
        }}
        .actions a.secondary {{
          background: var(--accent-2);
        }}
        @media (max-width: 900px) {{
          .hero, .plots {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="hero">
          <section class="panel">
            <span class="pill">Laverton WA</span>
            <span class="pill">Granny Smith / Wallaby Corridors</span>
            <span class="pill">Prototype Decision Support</span>
            <h1>Mining Risk Intelligence Dashboard</h1>
            <p class="sub">
              This local dashboard summarizes the current geotechnical edge-AI prototype:
              WA public regional context, synthetic operational proxies, model validation,
              and screening outputs for rapid engineering review.
            </p>
            <div class="metrics">
              <div class="metric">
                <div class="label">Selected model</div>
                <div class="value">{selected_model}</div>
              </div>
              <div class="metric">
                <div class="label">Accuracy</div>
                <div class="value">{selected_accuracy:.3f}</div>
              </div>
              <div class="metric">
                <div class="label">Macro F1</div>
                <div class="value">{selected_macro_f1:.3f}</div>
              </div>
              <div class="metric">
                <div class="label">White Reality Check p-value</div>
                <div class="value">{white_p}</div>
              </div>
            </div>
            <div class="actions">
              <a href="/docs">API Docs</a>
              <a class="secondary" href="https://github.com/jeevanreddy23/mining-risk-intelligence" target="_blank">GitHub Repo</a>
            </div>
          </section>

          <section class="panel">
            <h2>Model Comparison</h2>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Weighted F1</th>
                  <th>Macro F1</th>
                </tr>
              </thead>
              <tbody>
                {model_rows}
              </tbody>
            </table>
            <h3 style="margin-top: 24px;">Overfitting Check</h3>
            <p class="sub">{white_text}</p>
          </section>
        </div>

        <section>
          <h2>Pre-training and Post-training Visuals</h2>
          <div class="plots">
            {plot_cards}
          </div>
        </section>
      </div>
    </body>
    </html>
    """
