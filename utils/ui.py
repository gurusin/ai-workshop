import streamlit as st


def render_header(sub_title: str = "") -> None:
    crumb = (
        f' <span style="color:#3a6080;margin:0 6px;">／</span>'
        f'<span style="color:#7fb3d3;font-size:0.82rem;font-weight:400;">{sub_title}</span>'
        if sub_title else ""
    )
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);'
        f'margin:-4px -1rem 28px -1rem;padding:13px 28px;display:flex;align-items:center;">'
        f'<a href="?nav=home" style="color:white;text-decoration:none;font-size:1.05rem;'
        f'font-weight:800;letter-spacing:-0.3px;">🎮 Sudarshana\'s AI Playground</a>'
        f'{crumb}</div>',
        unsafe_allow_html=True,
    )


def score_bar(score: float, max_val: float = 10.0) -> str:
    pct = int((score / max_val) * 100)
    color = "#198754" if pct < 40 else "#ffc107" if pct < 70 else "#dc3545"
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="background:{color};width:{pct}%;"></div>'
        f'</div>'
    )
