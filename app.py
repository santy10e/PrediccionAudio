"""Interfaz web Streamlit del detector de anomalías acústicas.

Ejecutar con:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from audio_classifier import (
    AudioClip,
    AudioLoadError,
    IncompatibleModelError,
    ModelBundle,
    Prediction,
    charts,
    load_audio,
    train_model,
)
from audio_classifier.config import (
    MODEL_PATH,
    SAMPLES_DIR,
    SUPPORTED_EXTENSIONS,
    TRAIN_DIR,
)
from audio_classifier.dataset import discover_dataset, parse_filename
from audio_classifier.features import compute_mel_spectrogram_db, compute_mfcc
from audio_classifier.model import load_model

REPO_URL = "https://github.com/santy10e/PrediccionAudio"

st.set_page_config(
    page_title="Detector de Anomalías Acústicas",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------- estilos
st.markdown(
    """
<style>
:root {
  --good: #0ca30c; --good-bg: #ecf7ec; --good-ink: #006300;
  --critical: #d03b3b; --critical-bg: #fbeeee; --critical-ink: #a02a2a;
  --ink-1: #0b0b0b; --ink-2: #52514e; --ink-3: #898781;
  --hairline: rgba(11,11,11,0.10); --track: #e1e0d9;
}
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.app-header { display: flex; align-items: center; gap: .9rem; margin-bottom: .25rem; }
.app-header .logo {
  width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
  background: linear-gradient(135deg, #2a78d6, #184f95); color: #fff; font-size: 24px;
}
.app-header h1 { font-size: 1.65rem; margin: 0; padding: 0; }
.app-subtitle { color: var(--ink-2); margin: 0 0 1.25rem 0; font-size: .97rem; }

.verdict {
  border-radius: 14px; padding: 1.25rem 1.35rem; border: 1px solid var(--hairline);
  margin-bottom: .75rem;
}
.verdict.good { background: var(--good-bg); }
.verdict.critical { background: var(--critical-bg); }
.verdict .eyebrow {
  text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 600;
  color: var(--ink-2); margin-bottom: .35rem;
}
.verdict .status { display: flex; align-items: center; gap: .55rem; font-size: 1.7rem; font-weight: 700; }
.verdict.good .status { color: var(--good-ink); }
.verdict.critical .status { color: var(--critical-ink); }
.verdict .status .dot {
  width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
  color: #fff; font-size: 18px; font-weight: 700; flex: none;
}
.verdict.good .dot { background: var(--good); }
.verdict.critical .dot { background: var(--critical); }
.verdict .conf { color: var(--ink-2); margin-top: .2rem; font-size: .92rem; }
.verdict .conf b { color: var(--ink-1); }

.meter { margin-top: 1.1rem; }
.meter .row { display: flex; justify-content: space-between; font-size: .78rem; color: var(--ink-3); }
.meter .track {
  position: relative; height: 10px; border-radius: 6px; background: var(--track); margin: .35rem 0;
  overflow: visible;
}
.meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }
.meter .tick { position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: var(--ink-2); }
.meter .value { font-size: .82rem; color: var(--ink-2); }

.file-meta { color: var(--ink-3); font-size: .85rem; margin: -.35rem 0 .6rem 0; word-break: break-all; }
.empty {
  border: 1px dashed var(--track); border-radius: 14px; padding: 2.5rem 1.5rem; text-align: center;
  color: var(--ink-2);
}
.empty .big { font-size: 2.2rem; }
[data-testid="stMetricValue"] { font-size: 1.35rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------- modelo y cache
@st.cache_resource(show_spinner="Cargando modelo…")
def _cached_model(path: str, mtime: float) -> ModelBundle:
    return load_model(Path(path))


def get_model() -> tuple[ModelBundle | None, str | None]:
    """Devuelve ``(modelo, error)``. El caché se invalida si cambia el archivo."""
    if not MODEL_PATH.exists():
        return None, None
    try:
        return _cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime), None
    except IncompatibleModelError as exc:
        return None, str(exc)


@dataclass
class Analysis:
    clip: AudioClip
    prediction: Prediction
    mel_db: np.ndarray
    mfcc: np.ndarray


@st.cache_data(show_spinner=False, max_entries=64)
def analyze(audio_bytes: bytes, model_key: str, _bundle: ModelBundle) -> Analysis:
    cfg = _bundle.feature_config
    clip = load_audio(audio_bytes, cfg)
    mfcc = compute_mfcc(clip, cfg)
    return Analysis(
        clip=clip,
        prediction=_bundle.predict_features(mfcc.mean(axis=1).astype(np.float32)),
        mel_db=compute_mel_spectrogram_db(clip, cfg),
        mfcc=mfcc,
    )


@st.cache_data(show_spinner=False)
def dataset_summary() -> pd.DataFrame:
    try:
        files = discover_dataset(TRAIN_DIR)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Sección", "Normal", "Anomalía"])
    df = pd.DataFrame([{"Sección": f.section, "label": f.label} for f in files])
    table = df.pivot_table(index="Sección", columns="label", aggfunc="size", fill_value=0)
    table = table.reindex(columns=["normal", "anomaly"], fill_value=0)
    return table.rename(columns={"normal": "Normal", "anomaly": "Anomalía"}).reset_index()


def list_samples() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    exts = {f".{e}" for e in SUPPORTED_EXTENSIONS}
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in exts)


def model_key(bundle: ModelBundle) -> str:
    return bundle.trained_at


# ------------------------------------------------------------------------ componentes
def header() -> None:
    st.markdown(
        """
<div class="app-header"><div class="logo">〰</div><h1>Detector de Anomalías Acústicas</h1></div>
<p class="app-subtitle">Clasifica grabaciones de maquinaria industrial como <b>normales</b> o
<b>anómalas</b> a partir de sus coeficientes MFCC y un modelo Random Forest.</p>
""",
        unsafe_allow_html=True,
    )


def verdict_card(pred: Prediction) -> None:
    kind = "critical" if pred.is_anomaly else "good"
    icon = "!" if pred.is_anomaly else "✓"
    label = "Anomalía detectada" if pred.is_anomaly else "Funcionamiento normal"
    pct = pred.anomaly_probability * 100
    fill = "var(--critical)" if pred.is_anomaly else "var(--good)"
    st.markdown(
        f"""
<div class="verdict {kind}" role="status" aria-label="{label}">
  <div class="eyebrow">Resultado</div>
  <div class="status"><span class="dot" aria-hidden="true">{icon}</span>{label}</div>
  <div class="conf">Confianza del modelo: <b>{pred.confidence:.0%}</b></div>
  <div class="meter">
    <div class="row"><span>Normal</span><span>Umbral 50 %</span><span>Anomalía</span></div>
    <div class="track" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct:.0f}"
         aria-label="Probabilidad de anomalía">
      <div class="fill" style="width:{pct:.1f}%; background:{fill};"></div>
      <div class="tick"></div>
    </div>
    <div class="value">Probabilidad de anomalía: <b>{pct:.1f} %</b></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty"><div class="big">{icon}</div><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def require_model(bundle: ModelBundle | None, error: str | None) -> bool:
    if bundle is not None:
        return True
    if error:
        st.error(f"**El modelo no es compatible.** {error}", icon=":material/error:")
    else:
        st.warning(
            "**Aún no hay un modelo entrenado.** Ve a la pestaña *Modelo* para entrenarlo "
            "(≈30 s) o ejecuta `python -m audio_classifier.train`.",
            icon=":material/model_training:",
        )
    return False


def render_analysis(name: str, audio_bytes: bytes, bundle: ModelBundle) -> None:
    try:
        with st.spinner("Analizando audio…"):
            result = analyze(audio_bytes, model_key(bundle), bundle)
    except AudioLoadError as exc:
        st.error(f"**No se pudo procesar `{name}`.** {exc}", icon=":material/error:")
        return

    clip, pred = result.clip, result.prediction
    known = parse_filename(Path(name))

    left, right = st.columns([5, 7], gap="large")
    with left:
        verdict_card(pred)
        if known is not None:
            matches = known.label == pred.label
            st.caption(
                f"{':material/check_circle:' if matches else ':material/cancel:'} "
                f"Etiqueta según el nombre del archivo: **{charts.LABEL_ES[known.label]}** "
                f"(sección {known.section}) — la predicción "
                f"{'coincide' if matches else 'no coincide'}."
            )
    with right:
        st.markdown(f"##### {name}")
        st.audio(audio_bytes)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duración", f"{clip.duration:.2f} s")
        m2.metric("Frecuencia orig.", f"{clip.original_sample_rate / 1000:g} kHz")
        m3.metric("Canales", "Mono" if clip.original_channels == 1 else f"{clip.original_channels} → mono")
        m4.metric("Nivel RMS", f"{20 * np.log10(max(clip.rms, 1e-9)):.1f} dBFS")

    st.markdown("#### Visualización de la señal")
    hop = bundle.feature_config.hop_length
    t1, t2, t3 = st.tabs(
        [
            ":material/show_chart: Forma de onda",
            ":material/gradient: Espectrograma Mel",
            ":material/grid_on: MFCC",
        ]
    )
    with t1:
        st.plotly_chart(charts.waveform(clip), width="stretch", config={"displaylogo": False})
        st.caption(
            "Envolvente de amplitud (máximo y mínimo por intervalo) tras convertir a mono "
            f"y remuestrear a {clip.sample_rate / 1000:g} kHz."
        )
    with t2:
        st.plotly_chart(
            charts.mel_spectrogram(result.mel_db, clip, hop), width="stretch", config={"displaylogo": False}
        )
        st.caption("Energía por banda Mel a lo largo del tiempo, en dB relativos al máximo.")
    with t3:
        st.plotly_chart(
            charts.mfcc_heatmap(result.mfcc, clip, hop), width="stretch", config={"displaylogo": False}
        )
        st.caption(
            "Coeficientes MFCC normalizados por fila. El modelo usa la media temporal "
            f"de los {bundle.feature_config.n_mfcc} coeficientes."
        )


# ----------------------------------------------------------------------------- pestañas
def tab_analyze(bundle: ModelBundle | None, error: str | None) -> None:
    source = st.segmented_control(
        "Origen del audio",
        options=["upload", "sample"],
        format_func={
            "upload": ":material/upload: Subir archivo",
            "sample": ":material/library_music: Muestras incluidas",
        }.get,
        default="upload",
        required=True,
        label_visibility="collapsed",
    )

    name: str | None = None
    audio_bytes: bytes | None = None
    if source == "sample":
        samples = list_samples()
        if not samples:
            st.info("No hay muestras en `data/samples/`.", icon=":material/info:")
        else:
            choice = st.selectbox("Muestra", samples, format_func=lambda p: p.name)
            name, audio_bytes = choice.name, choice.read_bytes()
    else:
        uploaded = st.file_uploader(
            "Archivo de audio",
            type=list(SUPPORTED_EXTENSIONS),
            help="Formatos admitidos: WAV, FLAC y OGG. Cualquier frecuencia de muestreo; "
            "el audio estéreo se convierte a mono.",
        )
        if uploaded is not None:
            name, audio_bytes = uploaded.name, uploaded.getvalue()

    if audio_bytes is None:
        empty_state(
            "🎧",
            "Selecciona un audio para comenzar",
            "Sube una grabación o elige una de las muestras incluidas.",
        )
        return
    if require_model(bundle, error):
        render_analysis(name, audio_bytes, bundle)


def tab_batch(bundle: ModelBundle | None, error: str | None) -> None:
    st.markdown("Analiza varios archivos a la vez y exporta los resultados en CSV.")
    files = st.file_uploader(
        "Archivos de audio",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="batch_uploader",
    )
    if not files:
        empty_state("🗂️", "Sin archivos", "Arrastra uno o más archivos de audio para analizarlos en lote.")
        return
    if not require_model(bundle, error):
        return

    rows = []
    progress = st.progress(0.0, text="Procesando…")
    for i, f in enumerate(files, start=1):
        row = {"Archivo": f.name}
        try:
            res = analyze(f.getvalue(), model_key(bundle), bundle)
            row.update(
                {
                    "Resultado": "⚠️ Anomalía" if res.prediction.is_anomaly else "✅ Normal",
                    "P(anomalía)": res.prediction.anomaly_probability,
                    "Confianza": res.prediction.confidence,
                    "Duración (s)": round(res.clip.duration, 2),
                    "Frecuencia (Hz)": res.clip.original_sample_rate,
                    "Error": "",
                }
            )
        except AudioLoadError as exc:
            row.update({"Resultado": "❌ Error", "Error": str(exc)})
        rows.append(row)
        progress.progress(i / len(files), text=f"Procesando {i}/{len(files)}…")
    progress.empty()

    df = pd.DataFrame(rows)
    n_anom = int(df["Resultado"].str.contains("Anomalía").sum())
    n_err = int(df["Resultado"].str.contains("Error").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Archivos", len(df))
    c2.metric("Normales", len(df) - n_anom - n_err)
    c3.metric("Anomalías", n_anom)
    c4.metric("Errores", n_err)

    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "P(anomalía)": st.column_config.ProgressColumn(
                "P(anomalía)", format="percent", min_value=0.0, max_value=1.0
            ),
            "Confianza": st.column_config.NumberColumn(format="percent"),
            "Error": st.column_config.TextColumn(width="medium"),
        },
    )
    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"analisis_audio_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def tab_model(bundle: ModelBundle | None, error: str | None) -> None:
    info, action = st.columns([3, 1], vertical_alignment="center")
    with info:
        st.markdown(
            "Random Forest (300 árboles, clases balanceadas) sobre la media de 40 MFCC. "
            "Las métricas se calculan en un 20 % reservado, estratificado por sección y clase; "
            "el modelo final se reentrena con todos los datos."
        )
    with action:
        retrain = st.button(
            "Reentrenar modelo" if bundle else "Entrenar modelo",
            type="primary",
            icon=":material/model_training:",
            width="stretch",
        )

    if retrain:
        with st.status("Entrenando modelo…", expanded=True) as status:
            bar = st.progress(0.0, text="Extrayendo características…")
            try:
                new = train_model(
                    progress=lambda d, t: bar.progress(d / t, text=f"Extrayendo características {d}/{t}")
                )
                status.update(label="Ajustando Random Forest…")
                new.save()
            except (FileNotFoundError, ValueError, OSError) as exc:
                status.update(label="El entrenamiento falló", state="error")
                st.error(str(exc))
                return
            status.update(label="Modelo entrenado y guardado", state="complete", expanded=False)
        analyze.clear()
        st.toast("Modelo actualizado", icon=":material/check_circle:")
        st.rerun()

    if error:
        st.error(error, icon=":material/error:")
    if bundle is None:
        n_files = int(dataset_summary()[["Normal", "Anomalía"]].to_numpy().sum())
        empty_state(
            "🧠",
            "Sin modelo entrenado",
            f"Entrena el modelo con los {n_files} audios etiquetados de <code>data/source_test/</code>.",
        )
        return

    r = bundle.report
    c = st.columns(5)
    c[0].metric("Accuracy", f"{r.accuracy:.1%}")
    c[1].metric("Balanced acc.", f"{r.balanced_accuracy:.1%}")
    c[2].metric("Precisión", f"{r.precision:.1%}", help="De lo marcado como anomalía, cuánto lo era.")
    c[3].metric("Recall", f"{r.recall:.1%}", help="De las anomalías reales, cuántas se detectaron.")
    c[4].metric("ROC AUC", f"{r.roc_auc:.3f}")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Matriz de confusión")
        st.plotly_chart(
            charts.confusion_matrix(r.confusion_matrix), width="stretch", config={"displayModeBar": False}
        )
    with right:
        st.markdown("##### Accuracy por sección")
        sec = r.per_section_accuracy
        st.plotly_chart(
            charts.horizontal_bars(
                list(sec), list(sec.values()), ".1%", "Accuracy", height=300, xrange=(0, 1)
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Coeficientes más influyentes")
        st.plotly_chart(
            charts.feature_importance(r.feature_importances, bundle.feature_config.n_mfcc),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.markdown("##### Datos de entrenamiento")
        st.dataframe(dataset_summary(), hide_index=True, width="stretch")
        st.caption(
            f"{r.n_samples} audios · evaluación sobre {r.n_test} · "
            f"entrenado el {datetime.fromisoformat(bundle.trained_at):%d/%m/%Y %H:%M} UTC · "
            f"scikit-learn {bundle.sklearn_version}"
        )


# ---------------------------------------------------------------------------- sidebar
def sidebar(bundle: ModelBundle | None) -> None:
    with st.sidebar:
        st.markdown("### :material/graphic_eq: Audio Anomaly")
        with st.container(border=True):
            if bundle:
                st.markdown(":green-badge[:material/check_circle: Modelo listo]")
                st.markdown(
                    f"**ROC AUC** {bundle.report.roc_auc:.3f}  \n"
                    f"**Accuracy** {bundle.report.accuracy:.1%}  \n"
                    f"**Muestras** {bundle.report.n_samples}"
                )
            else:
                st.markdown(":orange-badge[:material/warning: Sin modelo]")
                st.caption("Entrénalo desde la pestaña *Modelo*.")

        st.markdown("##### Cómo funciona")
        st.markdown(
            "1. El audio se convierte a **mono, 16 kHz**.\n"
            "2. Se calculan **40 MFCC** y se promedian en el tiempo.\n"
            "3. Un **Random Forest** estima la probabilidad de anomalía.\n"
            "4. Se clasifica como anómalo si supera el **50 %**."
        )
        st.divider()
        st.caption(f"Datos: grabaciones de maquinaria con formato DCASE · [Código fuente]({REPO_URL})")


# ------------------------------------------------------------------------------- main
def main() -> None:
    bundle, error = get_model()
    sidebar(bundle)
    header()
    t_analyze, t_batch, t_model = st.tabs(
        [
            ":material/troubleshoot: Analizar",
            ":material/folder_open: Análisis por lotes",
            ":material/insights: Modelo",
        ]
    )
    with t_analyze:
        tab_analyze(bundle, error)
    with t_batch:
        tab_batch(bundle, error)
    with t_model:
        tab_model(bundle, error)


main()
