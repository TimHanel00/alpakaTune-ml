"""Static plots plus a small browser dashboard for prepared dataset surfaces."""

from __future__ import annotations

from collections import defaultdict
import html
from pathlib import Path
import re
from typing import Any

from .dataset import load_split_manifest
from .evaluation import NumpyEnsemble


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_split(split_manifest: Path, output: Path, artifact: Path | None = None) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exception:
        raise RuntimeError("plotting requires matplotlib; install requirements-dev.txt") from exception
    manifest, rows = load_split_manifest(split_manifest)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["surface_id"]].append(row)
    ensemble = NumpyEnsemble(artifact) if artifact else None
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for surface, surface_rows in sorted(grouped.items()):
        surface_rows.sort(key=lambda row: row["candidate_index"])
        figure, axis = plt.subplots(figsize=(10, 5.5))
        indexes = [row["candidate_index"] for row in surface_rows]
        runtimes = [row["runtime_seconds"] * 1.0e6 for row in surface_rows]
        axis.scatter(indexes, runtimes, s=8, alpha=0.55, label="Exhaustive measurements")
        if ensemble is not None:
            predictions, _uncertainty, _embedding = ensemble.predict(surface_rows)
            predicted_order = predictions.argsort()
            axis.plot(
                [indexes[index] for index in predicted_order],
                [runtimes[index] for index in predicted_order],
                linewidth=1.0,
                alpha=0.7,
                label="Model ranking",
            )
        axis.set_title(f"{surface} — {surface_rows[0]['device_id']}")
        axis.set_xlabel("Candidate index")
        axis.set_ylabel("Runtime (µs)")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        path = output / f"{_safe(surface)}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        images.append(path)
    buttons = "\n".join(
        f'<button type="button" data-image="{html.escape(path.name)}">{html.escape(path.stem)}</button>'
        for path in images
    )
    first = html.escape(images[0].name) if images else ""
    dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>alpakaTune ML surfaces</title>
<style>body{{font:16px system-ui;margin:2rem;background:#f7f8fa;color:#20242a}}button{{margin:.2rem;padding:.55rem .8rem}}img{{display:block;max-width:100%;margin-top:1rem;background:white;border:1px solid #ccd1d8}}</style></head>
<body><h1>{html.escape(manifest['split'].title())} exhaustive surfaces</h1>
<p>Switch between complete measured surfaces. Lines, when present, show model ranking rather than best-so-far projection.</p>
<div>{buttons}</div><img id="plot" src="{first}" alt="Selected tuning surface">
<script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.querySelector('#plot').src=b.dataset.image);</script>
</body></html>"""
    (output / "index.html").write_text(dashboard, encoding="utf-8")
    return images

