"""Offline static visualizations (matplotlib) for the anchor analysis.

Produces three figures that work fully offline:

1. ``routes_and_anchors.png`` — memory (purple) and test (teal) routes on a
   local metric plane with anchor markers.
2. ``follow_metrics.png`` — test-vs-memory follow metrics over route progress
   (cross-track error, heading error, visual similarity) with anchor ticks.
3. ``commands.png`` — a timeline of emitted clock-direction commands for both
   trajectories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_MEMORY = "#7c3aed"
COLOR_TEST = "#0d9488"
COLOR_ANCHOR = "#dc2626"


def render(memory, test, metrics, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _routes_and_anchors(memory, test, out / "routes_and_anchors.png")
    _follow_metrics(test, metrics, out / "follow_metrics.png")
    _commands(memory, test, out / "commands.png")


def _routes_and_anchors(memory, test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for session, color in ((memory, COLOR_MEMORY), (test, COLOR_TEST)):
        east = [p.east for p in session.keypoints]
        north = [p.north for p in session.keypoints]
        ax.plot(east, north, "-", color=color, linewidth=2, label=f"{session.name} route")
        for anchor in session.anchors:
            ax.plot(anchor.east, anchor.north, marker="*", color=COLOR_ANCHOR, markersize=14)
            ax.annotate(anchor.anchor_id, (anchor.east, anchor.north), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="black")
    ax.set_aspect("equal")
    ax.set_title("Memory (reference) vs Test trajectory with adaptive anchors")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _follow_metrics(test, metrics, out: Path) -> None:
    s = [m.ref_s_m for m in metrics]
    cross = [m.cross_track_error_m for m in metrics]
    heading = [abs(m.heading_error_deg) for m in metrics]
    visual = [m.visual_sim if m.visual_sim is not None else np.nan for m in metrics]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(s, cross, color=COLOR_TEST)
    axes[0].set_ylabel("cross-track (m)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(s, heading, color=COLOR_TEST)
    axes[1].set_ylabel("heading error (deg)")
    axes[1].grid(alpha=0.3)
    axes[2].plot(s, visual, color=COLOR_TEST)
    axes[2].set_ylabel("visual sim")
    axes[2].set_xlabel("reference progress (m)")
    axes[2].grid(alpha=0.3)
    for anchor in test.anchors:
        for axis in axes:
            axis.axvline(anchor.s_m, color=COLOR_ANCHOR, linestyle=":", alpha=0.5)
    fig.suptitle("Test trajectory follow capability vs memory reference")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _commands(memory, test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    y = {memory.name: 1, test.name: 0}
    for session, color in ((memory, COLOR_MEMORY), (test, COLOR_TEST)):
        for command in session.commands:
            turn = command.action == "turn"
            ax.barh(y[session.name], 1.0, left=command.epoch_s, height=0.6,
                    color=COLOR_ANCHOR if turn else color, alpha=0.85)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([test.name, memory.name])
    ax.set_xlabel("epoch time (s)")
    ax.set_title("Emitted clock-direction commands (red = turning commands)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)