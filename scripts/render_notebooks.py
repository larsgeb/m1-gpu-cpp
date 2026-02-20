#!/usr/bin/env python3
"""
In-process notebook renderer for m1-gpu-cpp demos.

Executes jupytext .py notebooks by running cells with exec() in the same
Python process, avoiding Jupyter kernel subprocess issues on macOS CI where
Metal GPU access is available to the main process but not reliably inherited
by kernel subprocesses.

Matplotlib figures are captured via a patched plt.show() and appended to
cell outputs as base64-encoded PNG images.  nbconvert's HTMLExporter is then
called as a library (no subprocess) to produce the final HTML.

Usage:
    python scripts/render_notebooks.py <output_dir> <notebook.py> [...]
"""

import base64
import io
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jupytext
import nbformat
from nbformat import v4
from nbconvert.exporters import HTMLExporter


# ---------------------------------------------------------------------------
# Figure capture helpers
# ---------------------------------------------------------------------------


def _capture_and_close() -> list[str]:
    """Save every open matplotlib figure as a base64 PNG and close all."""
    images: list[str] = []
    for fnum in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(fnum).savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        images.append(base64.b64encode(buf.read()).decode())
    plt.close("all")
    return images


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------


def execute_notebook(py_path: Path, out_dir: Path) -> None:
    print(f"Executing {py_path} ...")
    nb = jupytext.read(str(py_path))

    # Shared namespace — persists across cells, just like a real kernel session.
    ns: dict = {"__name__": "__main__"}

    # Patch plt.show() so that cells which call it trigger figure capture
    # instead of trying to open a GUI window.  Images are accumulated per-cell.
    cell_images: list[str] = []

    def _patched_show(*_args, **_kwargs) -> None:
        cell_images.extend(_capture_and_close())

    plt.show = _patched_show

    for exec_count, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue

        # Drop IPython magic lines that jupytext stores as comments.
        src_lines = [
            line
            for line in cell.source.splitlines()
            if not line.lstrip().startswith(("# %", "# %%"))
        ]
        src = "\n".join(src_lines).strip()

        cell.outputs = []
        cell.execution_count = exec_count

        if not src:
            continue

        cell_images.clear()
        stdout_buf = io.StringIO()

        try:
            sys.stdout = stdout_buf
            exec(compile(src, str(py_path), "exec"), ns)  # noqa: S102
        except Exception as exc:
            cell.outputs.append(
                v4.new_output(
                    output_type="error",
                    ename=type(exc).__name__,
                    evalue=str(exc),
                    traceback=traceback.format_exc().splitlines(),
                )
            )
        finally:
            sys.stdout = sys.__stdout__

        text = stdout_buf.getvalue()
        if text:
            cell.outputs.append(
                v4.new_output(output_type="stream", name="stdout", text=text)
            )

        # Capture any figures not yet collected by patched show().
        cell_images.extend(_capture_and_close())

        for img_b64 in cell_images:
            cell.outputs.append(
                v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_b64, "text/plain": "<Figure>"},
                    metadata={"image/png": {"width": 650}},
                )
            )

    # Render to HTML using nbconvert as a library (no subprocess).
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    body, _ = exporter.from_notebook_node(nb)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{py_path.stem}.html"
    out_path.write_text(body, encoding="utf-8")
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <notebook.py> [...]", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    for nb_path in (Path(p) for p in sys.argv[2:]):
        execute_notebook(nb_path, out_dir)


if __name__ == "__main__":
    main()
