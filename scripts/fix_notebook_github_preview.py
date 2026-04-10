#!/usr/bin/env python3
"""Приводит bert-text-classifier.ipynb к виду, который GitHub и nbconvert принимают без ошибок.

Убирает: stream без name, виджеты, Colab intrinsic+json, тяжёлый Colab HTML у таблиц (оставляет text/plain).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.validator import validate

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "bert-text-classifier.ipynb"

WIDGET_MIMES = (
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
)

COLAB_MIMES = (
    "application/vnd.google.colaboratory.intrinsic+json",
)

# Слишком длинный HTML в выводах иногда роняет превью GitHub; text/plain обычно есть у тех же ячеек.
MAX_HTML_CHARS = 120_000


def _clean_data_mimes(data: dict) -> dict:
    for m in WIDGET_MIMES + COLAB_MIMES:
        data.pop(m, None)
    plain = data.get("text/plain")
    html = data.get("text/html")
    if plain is not None and html is not None:
        hs = html if isinstance(html, str) else "".join(html)
        if "colab-df-container" in hs or len(hs) > MAX_HTML_CHARS:
            data.pop("text/html", None)
    return data


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else NOTEBOOK
    nb = nbformat.read(path, as_version=4)

    nb.metadata.pop("widgets", None)
    nb.metadata.pop("colab", None)

    for cell in nb.cells:
        cm = dict(cell.get("metadata") or {})
        for k in list(cm.keys()):
            if "widget" in k.lower():
                cm.pop(k, None)
        cm.pop("colab", None)
        cell["metadata"] = cm

        if cell.get("cell_type") != "code":
            continue

        ec = cell.get("execution_count")
        new_outputs = []
        for output in cell.get("outputs") or []:
            ot = output.get("output_type")

            if ot == "stream":
                output.pop("metadata", None)
                if "name" not in output:
                    output["name"] = "stdout"
                new_outputs.append(output)
                continue

            if ot in ("display_data", "execute_result"):
                data = _clean_data_mimes(dict(output.get("data") or {}))
                output["data"] = data
                if ot == "display_data" and not data:
                    continue
                if "metadata" not in output:
                    output["metadata"] = {}
                if ot == "execute_result" and "execution_count" not in output:
                    output["execution_count"] = ec
                om = dict(output.get("metadata") or {})
                om.pop("referenced_widgets", None)
                om.pop("colab", None)
                output["metadata"] = om
            elif ot == "error":
                pass
            else:
                om = output.get("metadata")
                if om:
                    om = dict(om)
                    om.pop("referenced_widgets", None)
                    om.pop("colab", None)
                    output["metadata"] = om

            new_outputs.append(output)

        cell["outputs"] = new_outputs

    # Явная схема, с которой хорошо дружит GitHub
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"].setdefault("display_name", "Python 3")
    nb.metadata["kernelspec"].setdefault("language", "python")
    nb.metadata["kernelspec"].setdefault("name", "python3")

    nbformat.write(nb, path)
    validate(nbformat.read(path, as_version=4))
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
