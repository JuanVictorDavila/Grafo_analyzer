import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import datetime
import pandas as pd

from engine import analyze_signatures, generate_pdf_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GrafoAnalyzer PreciSion V2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 50%, #1a1f3c 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    border: 1px solid #3d4a8a;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(99,179,237,0.08) 0%, transparent 60%);
}
.main-header h1 { color: #e8ecf8; font-size: 2rem; font-weight:700; margin:0; }
.main-header p  { color: #8a96c0; margin: 6px 0 0; font-size: 1rem; }
.header-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid #63b3ed44;
    color: #63b3ed;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}

.lot-card {
    background: linear-gradient(145deg, #ffffff, #f7f9fc);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.verdict-authentic {
    background: linear-gradient(135deg, #1a3a2e, #1e4d38);
    border: 1px solid #38a16940;
    border-left: 4px solid #38a169;
    border-radius: 10px; padding: 16px 20px; color: #9ae6b4; margin: 8px 0;
}
.verdict-inconclusive {
    background: linear-gradient(135deg, #3a2e1a, #4d3d1e);
    border: 1px solid #d69e2e40;
    border-left: 4px solid #d69e2e;
    border-radius: 10px; padding: 16px 20px; color: #faf089; margin: 8px 0;
}
.verdict-fake {
    background: linear-gradient(135deg, #3a1a1a, #4d1e1e);
    border: 1px solid #e53e3e40;
    border-left: 4px solid #e53e3e;
    border-radius: 10px; padding: 16px 20px; color: #feb2b2; margin: 8px 0;
}
.verdict-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; }
.verdict-text  { font-size: 0.9rem; opacity: 0.9; }

.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #718096; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #2d3748; font-family: 'JetBrains Mono', monospace; }

.section-title {
    font-size: 1rem; font-weight: 700; color: #2d3748;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px; margin: 20px 0 12px;
}
.flag-item {
    background: #fff5f5; border: 1px solid #fed7d7;
    border-left: 3px solid #e53e3e;
    border-radius: 6px; padding: 8px 12px;
    font-size: 0.85rem; color: #742a2a; margin: 4px 0;
}
.img-caption {
    text-align: center; font-size: 0.72rem;
    color: #718096; margin-top: 4px; font-weight: 600;
}
.run-btn button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    color: white !important; font-weight: 700 !important;
    border-radius: 10px !important; height: 52px !important;
    font-size: 1rem !important;
}
.stDownloadButton button {
    background: linear-gradient(135deg, #6b46c1, #553c9a) !important;
    color: white !important; font-weight: 700 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "num_lots"   not in st.session_state: st.session_state.num_lots = 1
if "results"    not in st.session_state: st.session_state.results  = None
if "pdf_bytes"  not in st.session_state: st.session_state.pdf_bytes = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(uploaded_file.getvalue())
        return f.name

def to_rgb(img):
    if img is None: return None
    if len(img.shape) == 2: return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_img(img, caption="", bgr=True):
    if img is None: return
    st.image(to_rgb(img) if bgr else img, use_container_width=True)
    st.markdown(f'<div class="img-caption">{caption}</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 GrafoAnalyzer")
    st.markdown("**Versão:** PreciSion V2")
    st.markdown("---")
    st.markdown("### Como usar")
    st.markdown("""
1. Clique **Adicionar Lote** para cada assinatura que quer analisar (máx. 10)
2. Faça upload da **Assinatura Questionada** (a suspeita)
3. Faça upload de pelo menos **2 Amostras Autênticas** do mesmo autor
4. Clique **▶ Rodar Análise**
5. Veja os resultados na tela ou **gere o PDF**
    """)
    st.markdown("---")
    st.markdown("### Limiares de Decisão")
    st.markdown("""
| Probabilidade | Veredito |
|---|---|
| ≥ 70% | ✅ Autêntica |
| 35–70% | ⚠️ Inconclusiva |
| < 35% | ❌ Falsidade |
    """)
    st.markdown("---")
    st.markdown("### Algoritmos")
    st.markdown("""
- **ORB** (28%) — Keypoints de canto
- **HOG** (28%) — Gradientes orientados
- **ECC + SSIM** (28%) — Alinhamento geométrico
- **Grid Density** (16%) — Equilíbrio espacial
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="header-badge">FORENSE COMPUTACIONAL</div>
  <h1>🔬 GrafoAnalyzer PreciSion V2</h1>
  <p>Sistema de Análise Forense de Assinaturas Manuscritas — ORB · HOG · ECC · FFT · LBP · Fourier · Entropia de Shannon</p>
</div>
""", unsafe_allow_html=True)

# ── Lot controls ──────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([4, 1, 1])
with ctrl_col1:
    n = st.session_state.num_lots
    st.markdown(f"**{n} lote(s)** configurado(s) &nbsp;|&nbsp; máximo: 10", unsafe_allow_html=True)
with ctrl_col2:
    if st.button("➕ Adicionar Lote", disabled=(st.session_state.num_lots >= 10),
                 use_container_width=True):
        st.session_state.num_lots += 1
        st.session_state.results = None
        st.rerun()
with ctrl_col3:
    if st.button("➖ Remover Lote", disabled=(st.session_state.num_lots <= 1),
                 use_container_width=True):
        st.session_state.num_lots -= 1
        st.session_state.results = None
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Lot input sections ────────────────────────────────────────────────────────
lot_files = []
for i in range(st.session_state.num_lots):
    with st.expander(f"📁  LOTE #{i+1}", expanded=(i == 0)):
        col_q, col_s = st.columns([1, 2])

        with col_q:
            st.markdown("**🔍 Assinatura Questionada**")
            st.caption("Imagem suspeita/questionada para análise")
            q_file = st.file_uploader("Questionada", type=["png", "jpg", "jpeg"],
                                      key=f"q_{i}", label_visibility="collapsed")
            if q_file:
                st.image(q_file, caption="Questionada", use_container_width=True)

        with col_s:
            st.markdown("**📋 Amostras Autênticas** *(mínimo 2, máximo 3)*")
            st.caption("Assinaturas genuínas do mesmo autor para calibrar o padrão")
            s_cols = st.columns(3)
            s_files = []
            for j, sc in enumerate(s_cols):
                with sc:
                    s_file = st.file_uploader(f"Amostra {j+1}", type=["png", "jpg", "jpeg"],
                                              key=f"s_{i}_{j}", label_visibility="visible")
                    if s_file:
                        st.image(s_file, caption=f"Amostra {j+1}", use_container_width=True)
                    s_files.append(s_file)

        lot_files.append({"q": q_file, "s": s_files})

# ── Validate lots ─────────────────────────────────────────────────────────────
valid_lots = []
for i, lot in enumerate(lot_files):
    if lot["q"] is not None:
        samples = [s for s in lot["s"] if s is not None]
        if len(samples) >= 2:
            valid_lots.append({"idx": i, "q": lot["q"], "s": samples})

st.markdown("---")

if not valid_lots:
    st.info("ℹ️  Configure pelo menos **1 lote completo** (1 Questionada + 2 Amostras Autênticas) para iniciar a análise.")

# ── Run button ────────────────────────────────────────────────────────────────
st.markdown('<div class="run-btn">', unsafe_allow_html=True)
run_clicked = st.button("▶  RODAR ANÁLISE EM LOTE", disabled=(not valid_lots),
                        use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if run_clicked:
    st.session_state.results  = None
    st.session_state.pdf_bytes = None
    results = []
    temp_paths = []

    progress_bar = st.progress(0, text="Iniciando análise...")
    log_area     = st.empty()
    log_lines    = []

    def log_cb(msg):
        log_lines.append(msg)
        log_area.code("\n".join(log_lines[-20:]), language=None)

    for step, lot in enumerate(valid_lots):
        frac = step / len(valid_lots)
        progress_bar.progress(frac, text=f"Processando Lote #{lot['idx']+1}…")

        q_path = save_upload(lot["q"])
        temp_paths.append(q_path)
        s_paths = []
        for sf in lot["s"]:
            p = save_upload(sf)
            temp_paths.append(p)
            s_paths.append(p)

        resultado = analyze_signatures(q_path, s_paths, log_callback=log_cb)
        if "error" in resultado:
            log_cb(f"[FALHA] Lote #{lot['idx']+1}: {resultado['error']}")
        else:
            resultado["_lot_idx"] = lot["idx"]
            results.append(resultado)
            log_cb(f"[OK] Lote #{lot['idx']+1} → {resultado['probability_percentage']:.1f}%")

        progress_bar.progress((step + 1) / len(valid_lots),
                              text=f"Lote #{lot['idx']+1} concluído")

    for p in temp_paths:
        try: os.remove(p)
        except: pass

    log_area.empty()
    progress_bar.empty()
    st.session_state.results = results
    st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results

    st.markdown("## 📊 Resultados da Análise Forense")

    for res in results:
        lot_num = res["_lot_idx"] + 1
        prob    = res["probability_percentage"]
        z       = res["z_score"]

        if prob >= 70:
            verdict_class = "verdict-authentic"
            verdict_icon  = "✅"
            verdict_label = "AUTÊNTICA"
        elif prob >= 35:
            verdict_class = "verdict-inconclusive"
            verdict_icon  = "⚠️"
            verdict_label = "INCONCLUSIVA"
        else:
            verdict_class = "verdict-fake"
            verdict_icon  = "❌"
            verdict_label = "FALSIDADE"

        with st.expander(f"{verdict_icon}  Lote #{lot_num}  —  {prob:.1f}%  —  {verdict_label}", expanded=True):

            # Verdict banner
            st.markdown(f"""
            <div class="{verdict_class}">
              <div class="verdict-title">{verdict_icon} {verdict_label}</div>
              <div class="verdict-text">{res['conclusion']}</div>
            </div>""", unsafe_allow_html=True)

            # KPI metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown('<div class="metric-box">'
                    f'<div class="metric-label">Similaridade Base</div>'
                    f'<div class="metric-value">{res["mean_sample_similarity"]:.1%}</div>'
                    '</div>', unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-box">'
                    f'<div class="metric-label">Similaridade Questionada</div>'
                    f'<div class="metric-value">{res["questioned_similarity"]:.1%}</div>'
                    '</div>', unsafe_allow_html=True)
            with m3:
                st.markdown('<div class="metric-box">'
                    f'<div class="metric-label">Z-Score</div>'
                    f'<div class="metric-value">{z:+.2f}</div>'
                    '</div>', unsafe_allow_html=True)
            with m4:
                st.markdown('<div class="metric-box">'
                    f'<div class="metric-label">Probabilidade</div>'
                    f'<div class="metric-value">{prob:.1f}%</div>'
                    '</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Questionada images ──
            st.markdown('<div class="section-title">📸 Assinatura Questionada — Camadas de Análise</div>',
                        unsafe_allow_html=True)
            ic1 = st.columns(4)
            with ic1[0]: show_img(res["q_orig"],       "1. Original")
            with ic1[1]: show_img(res["q_red"],        "2. Traçado Biométrico")
            with ic1[2]: show_img(res["q_img"],        "3. Binarizado", bgr=False)
            with ic1[3]: show_img(res["q_rhythm"],     "4. Ritmo Gráfico")

            ic2 = st.columns(4)
            with ic2[0]: show_img(res["q_hesitation"], "5. Pontos de Hesitação")
            with ic2[1]: show_img(res["q_skel"],       "6. Esqueleto Topológico")
            with ic2[2]: show_img(res["q_fft"],        "7. Espectro FFT")
            with ic2[3]: show_img(res["q_lbp"],        "8. Micro-textura LBP")

            # ── Sample images ──
            st.markdown('<div class="section-title">📋 Amostras Autênticas</div>',
                        unsafe_allow_html=True)
            n_s = len(res["s_imgs"])
            s_skels = res.get("s_skels", [None]*n_s)
            s_ffts  = res.get("s_ffts",  [None]*n_s)
            s_lbps  = res.get("s_lbps",  [None]*n_s)

            for j in range(n_s):
                st.markdown(f"*Amostra {j+1}:*")
                sc = st.columns(5)
                with sc[0]: show_img(res["s_origs"][j], "Original")
                with sc[1]: show_img(res["s_reds"][j],  "Traçado")
                with sc[2]: show_img(res["s_imgs"][j],  "Binarizado", bgr=False)
                with sc[3]: show_img(s_skels[j],        "Esqueleto")
                with sc[4]: show_img(s_ffts[j],         "FFT")

            # ── Graphotechnical metrics table ──
            metrics = res.get("summary_metrics", {})
            if metrics:
                st.markdown('<div class="section-title">📏 Métricas Grafotécnicas Comparadas</div>',
                            unsafe_allow_html=True)
                df = pd.DataFrame({
                    "Métrica":         ["Espessura (px)", "Inclinação Axial (°)", "Proporcionalidade",
                                        "Endpoints (E)", "Junctions (J)", "Energia FFT",
                                        "Entropia Pixel (nats)", "Entropia Gradientes (nats)",
                                        "Fourier Dist."],
                    "Questionada":     [f"{metrics['q_thickness']:.2f}",   f"{metrics['q_axial']:.1f}",
                                        f"{metrics['q_aspect']:.3f}",       f"{metrics['q_endpoints']:.0f}",
                                        f"{metrics['q_junctions']:.0f}",    f"{metrics['q_fft']:.2f}",
                                        f"{metrics.get('q_pixel_entropy',0):.3f}",
                                        f"{metrics.get('q_grad_entropy',0):.3f}",
                                        f"{metrics.get('q_fourier_dist',0):.3f}"],
                    "Base Autêntica":  [f"{metrics['s_thickness']:.2f}",   f"{metrics['s_axial']:.1f}",
                                        f"{metrics['s_aspect']:.3f}",       f"{metrics['s_endpoints']:.0f}",
                                        f"{metrics['s_junctions']:.0f}",    f"{metrics['s_fft']:.2f}",
                                        f"{metrics.get('s_pixel_entropy',0):.3f}",
                                        f"{metrics.get('s_grad_entropy',0):.3f}", "—"],
                })
                st.dataframe(df, hide_index=True, use_container_width=True)

            # ── Individual scores ──
            st.markdown('<div class="section-title">📐 Scores Individuais por Amostra</div>',
                        unsafe_allow_html=True)
            score_data = [{"Amostra": f"Amostra {k+1}",
                           "Score": f"{s:.2%}"}
                          for k, s in enumerate(res["q_scores"])]
            st.dataframe(pd.DataFrame(score_data), hide_index=True, use_container_width=True)

            # ── Forensic flags ──
            flags = res.get("advanced_flags", [])
            if flags:
                st.markdown(f'<div class="section-title">🚨 Alertas Forenses ({len(flags)} detectados)</div>',
                            unsafe_allow_html=True)
                for flag in flags:
                    st.markdown(f'<div class="flag-item">{flag}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ Nenhum alerta forense extramétrico detectado.")

    # ── Executive summary table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Resumo Executivo do Lote")
    summary_rows = []
    for res in results:
        p = res["probability_percentage"]
        if p >= 70:   v = "✅ AUTÊNTICA"
        elif p >= 35: v = "⚠️ INCONCLUSIVA"
        else:         v = "❌ FALSIDADE"
        summary_rows.append({
            "Lote":          f"#{res['_lot_idx']+1}",
            "Sim. Base":     f"{res['mean_sample_similarity']:.1%}",
            "Sim. Quest.":   f"{res['questioned_similarity']:.1%}",
            "Z-Score":       f"{res['z_score']:+.2f}",
            "Probabilidade": f"{p:.1f}%",
            "Veredito":      v,
            "Alertas":       len(res.get("advanced_flags", [])),
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # ── PDF generation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Relatório PDF Completo")
    st.caption("Gera o dossiê forense em PDF com todas as imagens, equações matemáticas e fundamentos científicos.")

    _pdf_needs_rerun = False

    if st.session_state.pdf_bytes is None:
        gen_col, _ = st.columns([1, 2])
        with gen_col:
            if st.button("🖨️  Gerar Laudo PDF", use_container_width=True, key="btn_gen_pdf"):
                _bar = st.progress(0, text="Preparando imagens temporárias…")
                try:
                    _bar.progress(20, text="Calculando páginas de análise…")
                    _pdf_raw = generate_pdf_report(results, return_bytes=True)
                    _bar.progress(90, text="Finalizando PDF…")
                    st.session_state.pdf_bytes = bytes(_pdf_raw)
                    _bar.progress(100, text="PDF gerado com sucesso!")
                    _pdf_needs_rerun = True
                except Exception as _e:
                    import traceback as _tb
                    _bar.empty()
                    st.error(f"❌ Erro ao gerar PDF: {_e}")
                    with st.expander("🔍 Traceback completo (para diagnóstico)"):
                        st.code(_tb.format_exc(), language="python")
    else:
        regen_col, _ = st.columns([1, 5])
        with regen_col:
            if st.button("🔄 Regenerar PDF", use_container_width=True, key="btn_regen_pdf"):
                st.session_state.pdf_bytes = None
                st.rerun()

    if st.session_state.pdf_bytes is not None:
        import io as _io
        _pdf_io = _io.BytesIO(st.session_state.pdf_bytes)
        _size_kb = len(st.session_state.pdf_bytes) / 1024
        fname = f"Laudo_GrafoAnalyzer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label=f"⬇️  Baixar Laudo PDF  ({_size_kb:.0f} KB)",
            data=_pdf_io,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            key="btn_download_pdf",
        )
        st.success(f"✅ PDF pronto — **{_size_kb:.0f} KB** · orientação paisagem · Clique acima para baixar.")

    if _pdf_needs_rerun:
        st.rerun()
