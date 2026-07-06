import streamlit as st

from config import APPS, CARD_ACCENT
from utils.ui import render_header


@st.dialog("App details", width="large")
def _show_app_modal(app_id: str) -> None:
    app = next((a for a in APPS if a["id"] == app_id), None)
    if not app:
        return
    tags_html = "".join(
        f'<span style="background:#f3f4f6;color:#374151;border-radius:20px;padding:3px 10px;'
        f'font-size:0.74rem;font-weight:600;">{t}</span>'
        for t in app["tags"]
    )
    accent = CARD_ACCENT.get(app["id"], "linear-gradient(90deg,#6366f1,#8b5cf6)")
    badge = (
        '<span style="background:#dcfce7;color:#166534;border:1px solid #86efac;border-radius:20px;'
        'padding:3px 10px;font-size:0.72rem;font-weight:700;">&#9679; LIVE</span>'
        if app["status"] == "live" else
        '<span style="background:#fef9c3;color:#854d0e;border:1px solid #fde68a;border-radius:20px;'
        'padding:3px 10px;font-size:0.72rem;font-weight:700;">&#9675; COMING SOON</span>'
    )
    st.markdown(
        f"""
        <div style="height:5px;border-radius:6px;background:{accent};margin-bottom:20px;"></div>
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
            <span style="font-size:2.8rem;line-height:1;">{app['icon']}</span>
            <div>
                {badge}
                <div style="font-size:1.25rem;font-weight:800;color:#111827;margin-top:6px;line-height:1.3;">{app['title']}</div>
            </div>
        </div>
        <p style="font-size:0.88rem;color:#374151;line-height:1.8;margin-bottom:18px;">{app['desc']}</p>
        <div style="background:#f0f9ff;border-left:4px solid #38bdf8;border-radius:0 8px 8px 0;padding:11px 15px;margin-bottom:18px;">
            <div style="font-size:0.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#0369a1;margin-bottom:4px;">Why it matters</div>
            <div style="font-size:0.85rem;color:#0369a1;line-height:1.65;">{app['why']}</div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:22px;">{tags_html}</div>
        """,
        unsafe_allow_html=True,
    )
    if app["status"] == "live":
        if st.button("▶ Launch App", type="primary", use_container_width=True, key=f"modal_launch_{app['id']}"):
            st.session_state["page"] = app["id"]
            st.session_state.pop("_modal_app", None)
            st.rerun()
    else:
        st.button("Coming Soon ⏳", disabled=True, use_container_width=True, key=f"modal_soon_{app['id']}")


def render_home() -> None:
    render_header()

    modal_id = st.session_state.pop("_modal_app", None)
    if modal_id:
        _show_app_modal(modal_id)

    st.markdown(
        "<p style='color:#6b7280;font-size:0.92rem;margin:-10px 0 32px;line-height:1.7;'>"
        "Where LangGraph agents scout live odds, LLMs referee promotional content, "
        "and responsible AI isn't just a slide &mdash; it <em>actually blocks things</em>. "
        "If something breaks mid-demo, we'll call it emergent behaviour."
        "</p>",
        unsafe_allow_html=True,
    )

    NCOLS = 4
    chunks = [APPS[i: i + NCOLS] for i in range(0, len(APPS), NCOLS)]

    st.markdown('<div class="section-label">Live demos</div>', unsafe_allow_html=True)

    for chunk in chunks:
        cols = st.columns(NCOLS, gap="medium")
        for idx in range(NCOLS):
            with cols[idx]:
                if idx >= len(chunk):
                    continue
                app = chunk[idx]
                accent = CARD_ACCENT.get(app["id"], "linear-gradient(90deg,#6366f1,#8b5cf6)")
                badge = (
                    '<span class="lp-badge-live">&#9679; LIVE</span>'
                    if app["status"] == "live" else
                    '<span class="lp-badge-soon">&#9675; SOON</span>'
                )
                tags_html = "".join(
                    f'<span class="lp-tag">{t}</span>'
                    for t in app["tags"][:4]
                )
                st.markdown(
                    f"""
                    <div class="lp-card">
                        <div class="lp-card-accent" style="background:{accent};"></div>
                        <div class="lp-card-body">
                            <div class="lp-card-top">
                                <span class="lp-card-icon">{app['icon']}</span>
                                {badge}
                            </div>
                            <div class="lp-card-title">{app['title']}</div>
                            <div class="lp-card-desc">{app['desc']}</div>
                            <div class="lp-card-tags">{tags_html}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                btn_cols = st.columns([1, 1], gap="small")
                with btn_cols[0]:
                    if st.button("Details →", key=f"more_{app['id']}", use_container_width=True):
                        st.session_state["_modal_app"] = app["id"]
                        st.rerun()
                with btn_cols[1]:
                    if app["status"] == "live":
                        if st.button("▶ Launch", key=f"nav_{app['id']}", type="primary", use_container_width=True):
                            st.session_state["page"] = app["id"]
                            st.rerun()
                    else:
                        st.button("Coming Soon", key=f"nav_{app['id']}", disabled=True, use_container_width=True)
                st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">Built with LangGraph &middot; XGBoost &middot; SHAP &middot; Streamlit &middot; Open-source LLMs</div>',
        unsafe_allow_html=True,
    )
