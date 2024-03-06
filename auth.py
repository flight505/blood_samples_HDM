import hmac
import os

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ⚠️ Instructions for using auth.py for authentication in a Streamlit app with a limited user base:
# 1. Create an .env file in the root directory of your application.
# 2. In that file, add users using the format: username='password' (one per line).
# 3. At the beginning of your app.py file (and all pages requiring authentication), insert these lines:
#    from auth import check_password
#    if not check_password():
#        st.stop()


# Ensure session state keys are initialized at the very beginning
def initialize_session_state():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "logged_in_user" not in st.session_state:
        st.session_state["logged_in_user"] = None


initialize_session_state()


# Your existing functions follow here
def check_password():
    """Check if the user has a correct password."""
    login_form()
    return st.session_state["password_correct"]


def login_form():
    """Display a form for user login in the sidebar."""
    with st.sidebar:
        if st.session_state["password_correct"]:
            st.write(f"Logged in as: `{st.session_state['logged_in_user']}`")
            if st.button("Logout"):
                st.session_state["password_correct"] = False
                st.session_state["logged_in_user"] = None
                st.rerun()  # Rerun the whole app.
        else:
            with st.form("Credentials"):
                st.text_input("Username", key="username")
                st.text_input("Password", type="password", key="password")
                submit_button = st.form_submit_button("Log in")

            if submit_button:
                password_entered()


def password_entered():
    """Validate the entered password."""
    username = st.session_state.get("username")
    entered_password = st.session_state.get("password", "")
    # Access expected password from Streamlit secrets
    expected_password = st.secrets["users"].get(username)

    if expected_password and hmac.compare_digest(entered_password, expected_password):
        st.session_state["password_correct"] = True
        st.session_state["logged_in_user"] = username
        st.toast(f"🔒 Logged in as: {username}")
        del st.session_state["password"]  # Don't store the password.
        st.rerun()  # Rerun the whole app.
    else:
        st.session_state["password_correct"] = False
        st.error("😕 User not known or password incorrect")
