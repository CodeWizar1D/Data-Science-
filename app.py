import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Configure full-width layout and browser metadata
st.set_page_config(
    page_title="DiabPredict — AI-Powered Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Seamless CSS: remove Streamlit chrome, paddings and headers
st.markdown(
    """
    <style>
        #MainMenu, header, footer {
            visibility: hidden;
            display: none !important;
        }
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }
        iframe {
            border: none !important;
            width: 100% !important;
            min-height: 100vh !important;
        }
        div[data-testid="stToolbar"] {
            display: none !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Path to the bundled self-contained frontend application
bundle_path = Path(__file__).resolve().parent / "dist" / "index_bundle.html"

if bundle_path.exists():
    html_content = bundle_path.read_text(encoding="utf-8")
    components.html(html_content, height=1200, scrolling=True)
else:
    st.error("Frontend build not found. Please ensure dist/index_bundle.html exists in the repository.")
