"""Gráficos Plotly con un estilo visual consistente."""

from __future__ import annotations

import librosa
import numpy as np
import plotly.graph_objects as go

from .audio_io import AudioClip
from .config import LABELS

# --- Tokens de color (paleta validada, modo claro) ---------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_1_FILL = "rgba(42, 120, 214, 0.18)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Rampa secuencial azul (cerca de cero = claro, magnitud alta = oscuro)
SEQUENTIAL_BLUE = [
    [0.0, "#f4f8fe"],
    [0.15, "#cde2fb"],
    [0.3, "#9ec5f4"],
    [0.45, "#6da7ec"],
    [0.6, "#3987e5"],
    [0.75, "#256abf"],
    [0.9, "#184f95"],
    [1.0, "#0d366b"],
]
# Divergente azul <-> rojo con punto medio gris neutro
DIVERGING = [
    [0.0, "#184f95"],
    [0.25, "#6da7ec"],
    [0.5, "#f0efec"],
    [0.75, "#ec8a89"],
    [1.0, "#b42e2e"],
]

LABEL_ES = {"normal": "Normal", "anomaly": "Anomalía"}


def _base_layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK_PRIMARY)),
        showlegend=False,
        **kwargs,
    )
    axis = dict(
        gridcolor=GRID,
        linecolor=AXIS,
        zeroline=False,
        tickfont=dict(color=INK_MUTED),
        title_font=dict(color=INK_SECONDARY, size=12),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def waveform(clip: AudioClip, buckets: int = 2000) -> go.Figure:
    """Envolvente min/max de la señal (rápida incluso para audios largos)."""
    sig = clip.signal
    n = max(1, len(sig) // buckets)
    usable = sig[: (len(sig) // n) * n].reshape(-1, n)
    upper, lower = usable.max(axis=1), usable.min(axis=1)
    t = (np.arange(len(upper)) * n + n / 2) / clip.sample_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=upper,
            mode="lines",
            line=dict(color=SERIES_1, width=1),
            name="Máx.",
            hovertemplate="%{x:.3f} s · máx %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=lower,
            mode="lines",
            line=dict(color=SERIES_1, width=1),
            fill="tonexty",
            fillcolor=SERIES_1_FILL,
            name="Mín.",
            hovertemplate="%{x:.3f} s · mín %{y:.4f}<extra></extra>",
        )
    )
    _base_layout(fig, 300, hovermode="x unified")
    fig.update_xaxes(title_text="Tiempo (s)", showspikes=True, spikecolor=AXIS, spikethickness=1)
    fig.update_yaxes(title_text="Amplitud", zeroline=True, zerolinecolor=AXIS)
    return fig


def _time_freq_heatmap(
    z: np.ndarray, x: np.ndarray, y: np.ndarray, colorscale, ytitle: str, zlabel: str, ytick: str, **heat
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(
                thickness=10,
                outlinewidth=0,
                tickfont=dict(color=INK_MUTED, size=11),
                title=dict(text=zlabel, font=dict(size=11, color=INK_SECONDARY)),
            ),
            hovertemplate=f"%{{x:.2f}} s · {ytick}<br>{zlabel}: %{{z:.2f}}<extra></extra>",
            **heat,
        )
    )
    _base_layout(fig, 320)
    fig.update_xaxes(title_text="Tiempo (s)", showgrid=False)
    fig.update_yaxes(title_text=ytitle, showgrid=False)
    return fig


def mel_spectrogram(mel_db: np.ndarray, clip: AudioClip, hop_length: int) -> go.Figure:
    frames = np.arange(mel_db.shape[1]) * hop_length / clip.sample_rate
    n_mels = mel_db.shape[0]
    hz = librosa.mel_frequencies(n_mels=n_mels, fmax=clip.sample_rate / 2)
    # Rango de color adaptado a la señal para no aplanar el contraste
    zmin = max(float(np.percentile(mel_db, 2)), -80.0)
    fig = _time_freq_heatmap(
        mel_db,
        frames,
        np.arange(n_mels),
        SEQUENTIAL_BLUE,
        "Frecuencia (Hz)",
        "dB",
        "%{customdata:.0f} Hz",
        zmin=zmin,
        zmax=0,
        customdata=np.repeat(hz[:, None], mel_db.shape[1], axis=1),
    )
    ticks = np.linspace(0, n_mels - 1, 6).round().astype(int)
    fig.update_yaxes(
        tickvals=ticks, ticktext=[f"{hz[i] / 1000:.1f} k" if hz[i] >= 1000 else f"{hz[i]:.0f}" for i in ticks]
    )
    return fig


def mfcc_heatmap(mfcc: np.ndarray, clip: AudioClip, hop_length: int) -> go.Figure:
    # Normalización por coeficiente para que todos sean comparables visualmente
    z = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    frames = np.arange(mfcc.shape[1]) * hop_length / clip.sample_rate
    return _time_freq_heatmap(
        z,
        frames,
        np.arange(1, mfcc.shape[0] + 1),
        DIVERGING,
        "Coeficiente MFCC",
        "z-score",
        "MFCC %{y}",
        zmid=0,
        zmin=-3,
        zmax=3,
    )


def confusion_matrix(cm: list[list[int]]) -> go.Figure:
    names = [LABEL_ES[lbl] for lbl in LABELS]
    z = np.array(cm)
    total = z.sum() or 1
    text = [[f"<b>{v}</b><br>{v / total:.0%}" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=names,
            y=names,
            colorscale=SEQUENTIAL_BLUE,
            showscale=False,
            xgap=2,
            ygap=2,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14),
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Casos: %{z}<extra></extra>",
        )
    )
    _base_layout(fig, 300)
    fig.update_xaxes(title_text="Predicción", side="bottom", showgrid=False)
    fig.update_yaxes(title_text="Valor real", autorange="reversed", showgrid=False)
    return fig


def horizontal_bars(
    labels: list[str],
    values: list[float],
    value_format: str,
    title: str,
    height: int | None = None,
    xrange: tuple[float, float] | None = None,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=SERIES_1, cornerradius=4),
            text=[format(v, value_format) for v in values],
            textposition="outside",
            textfont=dict(color=INK_SECONDARY),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:" + value_format + "}<extra></extra>",
        )
    )
    _base_layout(fig, height or max(200, 28 * len(labels) + 70), bargap=0.45)
    # Margen a la derecha para que las etiquetas exteriores no se recorten
    hi = xrange[1] if xrange else max(values, default=1)
    lo = xrange[0] if xrange else 0
    fig.update_xaxes(title_text=title, range=[lo, hi * 1.15], showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def feature_importance(importances: list[float], n_mfcc: int, top: int = 12) -> go.Figure:
    names = [f"MFCC {i + 1}" for i in range(n_mfcc)]
    order = np.argsort(importances)[::-1][:top]
    return horizontal_bars(
        [names[i] for i in order], [importances[i] for i in order], ".3f", "Importancia relativa"
    )
