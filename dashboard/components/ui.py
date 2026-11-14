"""Streamlit glue shared by all pages: page setup, friendly error boxes, data status."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from . import cached
from .paths import figures_root, models_root, results_root, runs_root


def setup(title: str, icon: str = ":material/precision_manufacturing:") -> None:
    """Page title and the sidebar data-status box (what is available in this checkout)."""
    try:
        st.set_page_config(page_title=f"{title} | RL Warehouse", page_icon=icon, layout="wide")
    except st.errors.StreamlitAPIException:
        pass  # already set (single-page tests, or called twice)
    st.title(title)
    with st.sidebar:
        st.caption("Data in this checkout")
        n_models = len(cached.models())
        n_groups = len(cached.run_groups())
        st.markdown(
            f"- `models/`: {n_models} curated model(s)\n"
            f"- `results/`: {'found' if results_root().is_dir() else 'missing'}\n"
            f"- `runs/`: {n_groups} run group(s)"
            + ("" if n_groups else " (not present, showing committed results only)")
        )


def notice(message: str, kind: str = "info") -> None:
    {"info": st.info, "warning": st.warning, "error": st.error}[kind](message)


def missing_file(path: Path | str, what: str = "file") -> None:
    st.info(f"The {what} `{_short(path)}` is not available in this checkout.")


def show_image(path: Path, caption: str | None = None, width: int | None = None) -> bool:
    """Show an image if it exists; otherwise a one-line note. Returns whether it was shown."""
    if not Path(path).is_file():
        missing_file(path, "image")
        return False
    if width:
        st.image(str(path), caption=caption, width=width)
    else:
        st.image(str(path), caption=caption)
    return True


def error_box(action: str, exc: Exception) -> None:
    """A short, readable error instead of a traceback."""
    st.error(f"Could not {action}: {type(exc).__name__}: {exc}")


def _short(path: Path | str) -> str:
    path = Path(path)
    for root in (runs_root(), models_root(), results_root(), figures_root()):
        try:
            return f"{root.name}/{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return path.name
