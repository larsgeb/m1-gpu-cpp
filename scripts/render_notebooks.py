#!/usr/bin/env python3
"""
In-process notebook renderer for m1-gpu-cpp demos.

PARENT MODE (default):
  For each notebook, spawns a fresh Python subprocess that runs all cells with
  exec() in-process, then writes the result as .ipynb.  Running each notebook
  in its own subprocess isolates any Metal ↔ matplotlib crash: if a subprocess
  exits non-zero (including SIGSEGV / exit 139), the parent catches it, writes
  a minimal error notebook, and continues to the next notebook.  HTML rendering
  runs in yet another subprocess (nbconvert) after all executions are done.

SUBPROCESS MODE (--execute-single <py> <ipynb>):
  Executes one notebook and writes the .ipynb.  Called by parent mode.
  Never invoked directly by users.

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
# Figure capture helpers
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
# Core executor  (runs inside the subprocess)
# ---------------------------------------------------------------------------


def execute_notebook(py_path: Path) -> nbformat.NotebookNode:
    """Execute a jupytext .py notebook in-process; return the notebook object."""
    print(f"Executing {py_path} ...", flush=True)
    nb = jupytext.read(str(py_path), fmt="py:light")

    ns: dict = {"__name__": "__main__"}
    plt_ref: list = []   # one-element list so the closure can rebind it
    cell_images: list[str] = []

    def _patched_show(*_args, **_kwargs) -> None:
        if plt_ref:
            cell_images.extend(_capture_and_close(plt_ref[0]))

    for exec_count, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue

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

        preview = src.splitlines()[0][:60]
        print(f"  [cell {exec_count}] {preview}", file=sys.__stdout__, flush=True)

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

        if not plt_ref and "plt" in ns:
            plt_ref.append(ns["plt"])
            plt_ref[0].show = _patched_show

        text = stdout_buf.getvalue()
        if text:
            cell.outputs.append(
                v4.new_output(output_type="stream", name="stdout", text=text)
            )

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

    return nb


# ---------------------------------------------------------------------------
# Subprocess entrypoint  (--execute-single mode)
# ---------------------------------------------------------------------------


def _run_single(py_path: Path, ipynb_path: Path) -> None:
    """Execute one notebook and write the .ipynb.  Called by the parent."""
    nb = execute_notebook(py_path)
    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ipynb_path, "w") as f:
        nbformat.write(nb, f)
    print(f"  saved {ipynb_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point  (parent / orchestrator mode)
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Subprocess mode: --execute-single <py_path> <ipynb_path>
    # ------------------------------------------------------------------
    if len(sys.argv) == 4 and sys.argv[1] == "--execute-single":
        _run_single(Path(sys.argv[2]), Path(sys.argv[3]))
        return

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <notebook.py> [...]", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    notebooks = [Path(p) for p in sys.argv[2:]]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-warm the matplotlib font cache.
    #   Building the cache takes 30-60 s on a fresh CI runner and happens
    #   inside `import matplotlib.pyplot as plt`.  When this coincides with
    #   Metal initialisation in the same subprocess the process can crash
    #   (SIGSEGV).  Running the import here — in the parent, which never
    #   touches Metal — writes the cache to disk so that every notebook
    #   subprocess finds it ready and skips the slow build entirely.
    # ------------------------------------------------------------------
    print("Pre-warming matplotlib font cache ...", flush=True)
    import matplotlib.pyplot as _plt  # noqa: PLC0415
    _plt.figure()
    _plt.close("all")
    del _plt
    print("  font cache ready.", flush=True)

    # ------------------------------------------------------------------
    # Phase 1 — execute each notebook in an isolated subprocess.
    #   If the subprocess crashes (e.g. SIGSEGV / exit 139), we write a
    #   minimal error notebook so that nbconvert always has something to
    #   convert and the CI job succeeds.
    # ------------------------------------------------------------------
    ipynb_files: list[Path] = []

    for nb_path in notebooks:
        ipynb_path = out_dir / f"{nb_path.stem}.ipynb"
        print(f"\n=== Rendering {nb_path} ===", flush=True)

        result = subprocess.run(
            [
                sys.executable, __file__,
                "--execute-single", str(nb_path), str(ipynb_path),
            ],
            capture_output=True,
            text=True,
            timeout=900,   # 15 min per notebook
        )

        # Always surface subprocess output in the CI log.
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)

        if result.returncode != 0:
            msg = (
                f"Subprocess exited with code {result.returncode}.\n\n"
                f"stderr (last 3 000 chars):\n{result.stderr[-3000:]}"
            )
            print(f"  WARNING: {nb_path} — {msg[:120]}", flush=True)

            # Create a fallback notebook with the error message so that
            # nbconvert still produces an HTML page.
            err_nb = v4.new_notebook()
            err_nb.cells.append(
                v4.new_markdown_cell(f"# Notebook rendering failed\n\n```\n{msg}\n```")
            )
            with open(ipynb_path, "w") as f:
                nbformat.write(err_nb, f)

        ipynb_files.append(ipynb_path)

    # ------------------------------------------------------------------
    # Phase 2 — convert to HTML in a subprocess (nbconvert isolated).
    # ------------------------------------------------------------------
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
