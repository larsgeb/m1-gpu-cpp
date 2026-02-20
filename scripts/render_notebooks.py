#!/usr/bin/env python3
"""
In-process notebook renderer for m1-gpu-cpp demos.

Executes jupytext .py notebooks by running cells with exec() in the same
Python process.  This avoids the Jupyter kernel subprocess which cannot
reliably access the Metal GPU on macOS CI runners.

Matplotlib figures are captured via a patched plt.show().  plt is obtained
lazily from the exec namespace after the notebook's own import cell runs, so
this script never imports matplotlib itself — the MPLBACKEND=Agg environment
variable (set in CI) is sufficient to control the backend.

HTML rendering is done by spawning a separate nbconvert subprocess *after* all
GPU work is finished, keeping nbconvert's heavy imports isolated from Metal.

Usage:
    MPLBACKEND=Agg python scripts/render_notebooks.py <output_dir> <nb.py> [...]
"""

import base64
import io
import subprocess
import sys
import traceback
from pathlib import Path

import jupytext
import nbformat
from nbformat import v4


# ---------------------------------------------------------------------------
# Figure capture helpers  (plt accessed through exec namespace, never imported
# at module level to avoid a matplotlib ↔ Metal conflict at initialisation)
# ---------------------------------------------------------------------------


def _capture_and_close(plt) -> list[str]:
    """Save every open matplotlib figure as base64 PNG and close all."""
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


def execute_notebook(py_path: Path, out_dir: Path) -> Path:
    """Execute a jupytext .py notebook in-process and save the result as .ipynb."""
    print(f"Executing {py_path} ...")
    nb = jupytext.read(str(py_path), fmt="py:light")

    # Shared namespace — persists across all cells, just like a real kernel.
    ns: dict = {"__name__": "__main__"}

    # plt is obtained from ns after the notebook's first import cell runs.
    # cell_images accumulates captured figures for the current cell.
    plt_ref: list = []   # one-element list so the closure can rebind it
    cell_images: list[str] = []

    def _patched_show(*_args, **_kwargs) -> None:
        if plt_ref:
            cell_images.extend(_capture_and_close(plt_ref[0]))

    for exec_count, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue

        # Strip IPython magic lines.  jupytext stores them as `# %magic`
        # comments; when converting to a notebook it removes the `# ` prefix,
        # producing bare `%magic` lines that are not valid Python.
        src_lines = [
            line
            for line in cell.source.splitlines()
            if not line.lstrip().startswith(("# %", "# %%", "%", "%%"))
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

        # Once plt appears in the namespace, patch show() so subsequent cells
        # that call plt.show() trigger figure capture instead of a GUI window.
        if not plt_ref and "plt" in ns:
            plt_ref.append(ns["plt"])
            plt_ref[0].show = _patched_show

        text = stdout_buf.getvalue()
        if text:
            cell.outputs.append(
                v4.new_output(output_type="stream", name="stdout", text=text)
            )

        # Capture any figures not yet collected by the patched show().
        if plt_ref:
            cell_images.extend(_capture_and_close(plt_ref[0]))

        for img_b64 in cell_images:
            cell.outputs.append(
                v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_b64, "text/plain": "<Figure>"},
                    metadata={"image/png": {"width": 650}},
                )
            )

    # Save the executed notebook.
    out_dir.mkdir(parents=True, exist_ok=True)
    ipynb_path = out_dir / f"{py_path.stem}.ipynb"
    with open(ipynb_path, "w") as f:
        nbformat.write(nb, f)
    print(f"  saved {ipynb_path}")
    return ipynb_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <notebook.py> [...]", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    notebooks = [Path(p) for p in sys.argv[2:]]

    # Phase 1 — execute all notebooks in-process (Metal GPU access required).
    ipynb_files: list[Path] = []
    for nb_path in notebooks:
        ipynb_files.append(execute_notebook(nb_path, out_dir))

    # Phase 2 — convert to HTML in a subprocess so that nbconvert's heavy
    # imports (jinja2, lxml, …) are fully isolated from the Metal runtime.
    for ipynb_path in ipynb_files:
        subprocess.run(
            [
                sys.executable, "-m", "nbconvert",
                "--to", "html",
                "--output-dir", str(out_dir),
                str(ipynb_path),
            ],
            check=True,
        )
        ipynb_path.unlink()
        print(f"  → {out_dir / ipynb_path.with_suffix('.html').name}")


if __name__ == "__main__":
    main()
