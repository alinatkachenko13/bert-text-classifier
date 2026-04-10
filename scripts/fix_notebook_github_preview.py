#!/usr/bin/env python3
"""Приводит bert-text-classifier.ipynb к виду, который GitHub и nbconvert принимают без ошибок.

После сохранения ноутбука из Colab часто пропадает поле name у stream-выводов и появляются
виджеты в нестандартном виде. Скрипт правит это in-place."""

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


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else NOTEBOOK
    nb = nbformat.read(path, as_version=4)

    nb.metadata.pop("widgets", None)

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        cm = dict(cell.get("metadata") or {})
        for k in list(cm.keys()):
            if "widget" in k.lower():
                cm.pop(k, None)
        cell["metadata"] = cm

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
                data = dict(output.get("data") or {})
                for m in WIDGET_MIMES:
                    data.pop(m, None)
                output["data"] = data
                if ot == "display_data" and not data:
                    continue
                if "metadata" not in output:
                    output["metadata"] = {}
                if ot == "execute_result" and "execution_count" not in output:
                    output["execution_count"] = ec
                om = dict(output.get("metadata") or {})
                om.pop("referenced_widgets", None)
                output["metadata"] = om
            elif ot == "error":
                pass
            else:
                om = output.get("metadata")
                if om:
                    om = dict(om)
                    om.pop("referenced_widgets", None)
                    output["metadata"] = om

            new_outputs.append(output)

        cell["outputs"] = new_outputs

    nbformat.write(nb, path)
    validate(nbformat.read(path, as_version=4))
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
