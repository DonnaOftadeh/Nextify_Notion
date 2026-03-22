from pathlib import Path
import streamlit as st

def show_home() -> None:
    """Render the static Nextify landing page with a single scrollbar."""
    st.set_page_config(page_title="Nextify", page_icon="🤖", layout="wide")

    html_file = Path(__file__).with_name("Nextify_home_page.html")
    if html_file.exists():
        st.components.v1.html(
            html_file.read_text(encoding="utf-8"),
            height=9000,        # adjust so the whole page fits
            scrolling=False     # only the Streamlit page scrolls
        )
    else:
        st.title("Nextify")
        st.error("Nextify_home_page.html not found")

if __name__ == "__main__":
    show_home()
