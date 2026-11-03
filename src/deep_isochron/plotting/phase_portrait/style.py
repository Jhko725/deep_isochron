"""The drawing constants of the phase-portrait animations: `PortraitStyle`, a frozen
dataclass changed one field at a time with `copy.replace`.

The design and its decisions are in `docs/changes/2026-11-02-ksiam-animations.md`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PortraitStyle:
    """The drawing constants of the phase-portrait animations; `copy.replace` changes
    one.

    **Attributes:**

    - `figsize`, `dpi`: The figure size in inches and its resolution. The defaults give
        1920 × 800 pixels, even in both dimensions as the `yuv420p` pixel format of an
        MP4 requires.
    - `figsize_square`: The figure size in inches of the one-panel animation
        `animate_phase_field`, square; 960 × 960 pixels at the default `dpi`.
    - `frames_per_period`: The frames drawn per period of the cycle; with `fps`, one
        period plays in 2 s.
    - `fps`: The playback rate in frames per second.
    - `hold_seconds`: How long the last frame is held, in seconds, with the Δφ labels
        of `animate_kick` or the φ labels of `animate_isochron_sync` shown.
    - `pause_seconds`: How long an animation freezes at its key instant, in seconds:
        on the first frame after the kick in `animate_kick`, with the kick arrows
        shown, and on the start in `animate_isochron_sync`.
    - `tail_periods`: The length of the fading tail behind each point in the phase
        plane, in periods.
    - `kick_arrow_frames`: The number of frames after the kick pause for which the kick
        arrows stay drawn.
    - `box`: The half-width of the square phase-plane view.
    - `font_size`: The size of all text, in points; the minimum readable on a slide.
    - `annotation_font_size`: The size of the Δφ labels of `animate_kick` and of the φ
        labels of `animate_isochron_sync`, in points.
    - `cycle_width`, `isochron_width`: The line widths of the limit cycle and of the
        isochrons.
    - `trace_width`: The line width of the tails and of the time series.
    - `point_size`: The marker size of each moving point.
    - `cycle_color`: The limit cycle.
    - `reference_color`: The unkicked reference point.
    - `delay_color`, `advance_color`: A kicked copy whose phase falls behind the
        reference's, and one whose phase moves ahead or stays level.
    - `group_colors`: The points seeded on the first and on the second isochron of
        `animate_isochron_sync`, with their isochrons and time series.
    - `isochron_color`: The isochrons drawn for orientation only.
    """

    figsize: tuple[float, float] = (12.0, 5.0)
    figsize_square: tuple[float, float] = (6.0, 6.0)
    dpi: int = 160
    frames_per_period: int = 40
    fps: int = 20
    hold_seconds: float = 3.0
    pause_seconds: float = 1.5
    tail_periods: float = 0.5
    kick_arrow_frames: int = 3
    box: float = 1.8
    font_size: float = 18.0
    annotation_font_size: float = 20.0
    cycle_width: float = 3.0
    isochron_width: float = 1.0
    trace_width: float = 2.0
    point_size: float = 9.0
    cycle_color: str = "black"
    reference_color: str = "#0072B2"
    delay_color: str = "#D55E00"
    advance_color: str = "#009E73"
    group_colors: tuple[str, str] = ("#0072B2", "#CC79A7")
    isochron_color: str = "0.85"
