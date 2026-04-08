import streamlit as st
import requests
import base64
import json
from dotenv import load_dotenv

load_dotenv()
API_URL = "http://localhost:8000"

# ─────────────────────────── API helpers ────────────────────────────

def get_headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def api_register(data):
    try:
        r = requests.post(f"{API_URL}/auth/register", json=data, timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": f"Cannot reach backend: {e}"}, 500

def api_login(data):
    try:
        r = requests.post(f"{API_URL}/auth/login", json=data, timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": f"Cannot reach backend: {e}"}, 500

def api_get_me():
    try:
        r = requests.get(f"{API_URL}/auth/me", headers=get_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def api_update_profile(data):
    try:
        r = requests.put(f"{API_URL}/auth/profile", json=data, headers=get_headers(), timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500

def api_upload(files):
    try:
        upload_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
        r = requests.post(f"{API_URL}/upload", files=upload_files, headers=get_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def api_ask(question, session_id, task_type, language):
    try:
        payload = {"question": question, "session_id": session_id,
                   "task_type": task_type, "language": language}
        r = requests.post(f"{API_URL}/ask", json=payload, headers=get_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"answer": f"Error: {e}"}

# ─────────────────────────── Theme CSS ────────────────────────────

def apply_theme(theme: str = "light"):
    is_dark = theme == "dark" or (
        theme == "system" and st.get_option("theme.base") == "dark"
    )
    bg        = "#0d1117" if is_dark else "#ffffff"
    sidebar   = "#161b22" if is_dark else "#f0f4f8"
    text      = "#e6edf3" if is_dark else "#0d1b2a"
    border    = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.08)"
    accent    = "#1E90FF"
    accent_dk = "#1565C0"
    card_bg   = "#21262d" if is_dark else "#f8fafc"
    input_bg  = "#1c2128" if is_dark else "#ffffff"
    msg_user  = "#1565C0"
    msg_bot   = "#21262d" if is_dark else "#f0f4f8"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        font-size: 15px;
    }}

    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {sidebar} !important;
        border-right: 1px solid {border};
    }}
    [data-testid="stSidebar"] * {{ color: {text} !important; }}

    /* ── Headings ── */
    h1 {{ font-size: 1.7rem !important; font-weight: 700 !important;
          background: linear-gradient(135deg, {accent}, {accent_dk});
          -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    h2, h3 {{ color: {accent} !important; font-weight: 600 !important; }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, {accent}, {accent_dk}) !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important; font-size: 12px !important;
        padding: 0.35rem 1.1rem !important;
        transition: all 0.2s ease !important; font-weight: 500 !important;
    }}
    .stButton > button:hover {{
        opacity: 0.88 !important; transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(30,144,255,0.4) !important;
    }}

    /* ── Inputs ── */
    .stTextInput > div > input, .stTextArea > div > textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important; font-size: 13px !important;
    }}
    .stSelectbox > div > div {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important; border-radius: 8px !important;
    }}

    /* ── Cards ── */
    .card {{
        background: {card_bg}; border: 1px solid {border};
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }}

    /* ── Profile chip ── */
    .profile-chip {{
        display: flex; align-items: center; gap: 10px;
        padding: 0.6rem 0.75rem; border-radius: 10px;
        background: rgba(30,144,255,0.08);
        border: 1px solid rgba(30,144,255,0.15);
        margin-bottom: 0.75rem;
    }}
    .avatar {{
        width: 36px; height: 36px; border-radius: 50%;
        object-fit: cover; border: 2px solid {accent};
        flex-shrink: 0;
    }}
    .avatar-placeholder {{
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, {accent}, {accent_dk});
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; flex-shrink: 0;
    }}
    .pname {{ font-weight: 600; font-size: 0.85rem; color: {text}; }}
    .pemail {{ font-size: 0.72rem; color: #888; }}

    /* ── Chat messages ── */
    .stChatMessage {{ border-radius: 10px !important; margin-bottom: 0.4rem; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(30,144,255,0.3); border-radius: 10px; }}

    /* ── Auth container ── */
    .auth-wrap {{
        background: {card_bg}; border: 1px solid {border};
        border-radius: 16px; padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }}
    .auth-logo {{
        font-size: 2rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, {accent}, {accent_dk});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }}
    .auth-sub {{ text-align: center; color: #888; font-size: 0.8rem; margin-bottom: 1.5rem; }}

    /* ── Divider ── */
    hr {{ border-color: {border} !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {card_bg} !important; border-radius: 10px;
        border: 1px solid {border};
    }}
    .stTabs [data-baseweb="tab"] {{ color: {text} !important; font-size: 12px; }}
    .stTabs [aria-selected="true"] {{ color: {accent} !important; font-weight: 600; }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{ border: 1px dashed {border} !important;
        border-radius: 10px !important; background: {card_bg} !important; }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────── Auth page ────────────────────────────

def render_auth_page():
    apply_theme("light")
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
        <div class="auth-wrap">
            <div class="auth-logo">📄 DocChat Pro</div>
            <div class="auth-sub">AI-powered PDF intelligence platform</div>
        </div>""", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            uname = st.text_input("Username", key="li_user", placeholder="your_username")
            pwd   = st.text_input("Password", type="password", key="li_pwd", placeholder="••••••••")
            if st.button("Sign In", use_container_width=True, key="btn_login"):
                if not uname or not pwd:
                    st.warning("Please enter username and password")
                else:
                    with st.spinner("Signing in..."):
                        resp, code = api_login({"username": uname, "password": pwd})
                    if code == 200:
                        st.session_state.token = resp["access_token"]
                        st.session_state.username = uname
                        me = api_get_me()
                        if me:
                            st.session_state.profile = me
                        st.success("Welcome back! 👋")
                        st.rerun()
                    else:
                        st.error(resp.get("detail", "Login failed"))

        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            full = st.text_input("Full Name",  key="rg_full", placeholder="John Doe")
            user = st.text_input("Username",   key="rg_user", placeholder="johndoe")
            mail = st.text_input("Email",      key="rg_mail", placeholder="john@gmail.com")
            pw1  = st.text_input("Password",   type="password", key="rg_pw1", placeholder="Min 6 chars")
            pw2  = st.text_input("Confirm Password", type="password", key="rg_pw2", placeholder="Repeat password")
            if st.button("Create Account", use_container_width=True, key="btn_reg"):
                if not all([full, user, mail, pw1, pw2]):
                    st.warning("Please fill in all fields")
                elif pw1 != pw2:
                    st.error("Passwords do not match")
                elif len(pw1) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    with st.spinner("Creating account..."):
                        resp, code = api_register({"full_name": full, "username": user,
                                                   "email": mail, "password": pw1})
                    if code == 200:
                        st.success("Account created! Please log in. ✅")
                    else:
                        st.error(resp.get("detail", "Registration failed"))

# ─────────────────────────── Profile Settings Overlay ────────────────────────────

def render_profile_settings(profile: dict):
    st.subheader("⚙️ Profile & Settings")
    st.divider()

    # Current avatar
    img_b64 = profile.get("profile_image")
    if img_b64:
        st.markdown(f'<img src="data:image/png;base64,{img_b64}" style="width:80px;height:80px;border-radius:50%;border:3px solid #1E90FF;display:block;margin:auto;">', unsafe_allow_html=True)
    else:
        st.markdown('<div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,#1E90FF,#1565C0);display:flex;align-items:center;justify-content:center;font-size:32px;margin:auto;">👤</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload profile photo
    new_img = st.file_uploader("📷 Change Profile Photo", type=["png", "jpg", "jpeg"], key="profile_photo_upload")

    # Name
    new_name = st.text_input("Display Name", value=profile.get("full_name", ""), key="pname_input")

    # Theme
    theme_map = {"☀️ Light": "light", "🌙 Dark": "dark", "💻 System": "system"}
    reverse_map = {v: k for k, v in theme_map.items()}
    current_theme_label = reverse_map.get(profile.get("theme", "light"), "☀️ Light")
    theme_choice_label = st.radio("🎨 Theme", list(theme_map.keys()),
                                  index=list(theme_map.keys()).index(current_theme_label),
                                  horizontal=True, key="theme_radio")
    chosen_theme = theme_map[theme_choice_label]

    st.divider()
    if st.button("💾 Save Settings", use_container_width=True, key="save_profile"):
        update_data = {}
        if new_name and new_name != profile.get("full_name"):
            update_data["full_name"] = new_name
        if chosen_theme != profile.get("theme"):
            update_data["theme"] = chosen_theme
        if new_img:
            img_bytes = new_img.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            update_data["profile_image"] = b64

        if update_data:
            with st.spinner("Saving..."):
                resp, code = api_update_profile(update_data)
            if code == 200:
                st.session_state.profile = resp
                st.success("Saved! ✅")
                st.rerun()
            else:
                st.error(resp.get("detail", "Save failed"))
        else:
            st.info("No changes to save.")

    if st.button("🚪 Logout", use_container_width=True, key="logout_profile"):
        for k in ["token", "username", "profile", "messages", "session_id", "filenames"]:
            st.session_state.pop(k, None)
        st.rerun()

# ─────────────────────────── Main App ────────────────────────────

def render_main_app(profile: dict):
    theme = profile.get("theme", "light")
    apply_theme(theme)

    # Sidebar
    with st.sidebar:
        # Profile chip
        img_b64 = profile.get("profile_image")
        if img_b64:
            avatar_html = f'<img src="data:image/png;base64,{img_b64}" class="avatar">'
        else:
            avatar_html = '<div class="avatar-placeholder">👤</div>'

        display_name = profile.get("full_name", st.session_state.get("username", "User"))
        email        = profile.get("email", "")

        st.markdown(f"""
        <div class="profile-chip">
            {avatar_html}
            <div>
                <div class="pname">{display_name}</div>
                <div class="pemail">@{st.session_state.get('username','')}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Settings toggle
        if st.button("⚙️ Profile Settings", use_container_width=True, key="toggle_settings"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)

        st.divider()

        # Document upload
        st.markdown("**📚 Documents**")
        files = st.file_uploader("Upload PDFs", type="pdf",
                                 accept_multiple_files=True, label_visibility="collapsed")
        if files:
            if st.button("🚀 Process Files", use_container_width=True, key="process_files"):
                with st.spinner("Building knowledge base..."):
                    res = api_upload(files)
                if res:
                    st.session_state.session_id = res["session_id"]
                    st.session_state.filenames  = res["filenames"]
                    st.success(f"✅ {len(res['filenames'])} file(s) ready")

        if st.session_state.get("filenames"):
            with st.expander("📁 Loaded files", expanded=False):
                for fn in st.session_state.filenames:
                    st.caption(f"• {fn}")

        st.divider()

        # Task configuration
        st.markdown("**⚙️ Task**")
        task_options = ["Auto-detect", "Summary", "Compare PDF", "JSON Format", "Documentation"]
        st.session_state.task_type = st.selectbox("Type", task_options, label_visibility="collapsed")

        st.divider()

        # Language — placed lower / below task
        st.markdown("**🌐 Response Language**")
        lang_options = ["English", "Arabic", "Hindi", "Bengali", "Spanish", "French", "German"]
        st.session_state.language = st.selectbox("Language", lang_options, label_visibility="collapsed")

        if st.session_state.get("messages"):
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    # ── Main area ──
    if st.session_state.get("show_settings", False):
        render_profile_settings(profile)
        return

    st.markdown(
        "<h1>📄 DocChat Pro</h1>"
        "<p style='color:#666;font-size:0.82rem;margin-top:-0.5rem;'>Upload PDFs in the sidebar, then ask your questions below.</p>",
        unsafe_allow_html=True
    )

    # Chat history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.get("session_id"):
                response_text = "⚠️ Please upload and process at least one PDF first."
            else:
                with st.spinner("Analyzing..."):
                    res = api_ask(
                        prompt,
                        st.session_state.session_id,
                        st.session_state.get("task_type", "Auto-detect"),
                        st.session_state.get("language", "English")
                    )
                    response_text = res.get("answer", "No response received.")
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# ─────────────────────────── Entry point ────────────────────────────

def main():
    st.set_page_config(page_title="DocChat Pro", page_icon="📄", layout="wide",
                       initial_sidebar_state="expanded")

    # Init session state
    for key, default in [("token", None), ("username", None), ("profile", None),
                          ("messages", []), ("session_id", None), ("filenames", []),
                          ("show_settings", False)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Load profile once per session
    if st.session_state.token and st.session_state.profile is None:
        me = api_get_me()
        if me:
            st.session_state.profile = me
        else:
            st.session_state.token = None  # token expired/invalid

    if not st.session_state.token:
        render_auth_page()
    else:
        profile = st.session_state.profile or {}
        render_main_app(profile)

if __name__ == "__main__":
    main()