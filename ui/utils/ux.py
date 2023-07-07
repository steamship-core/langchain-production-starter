import streamlit as st
from steamship import PackageInstance


def sidebar():
    with st.sidebar:
        api_key = st.text_input("Steamship API Key", value=st.session_state.get("steamship_api_key"), type="password")
        st.session_state.steamship_api_key = api_key
        if not st.session_state.get("steamship_api_key"):
            st.write("[Click here to get your free key](https://www.steamship.com/account/api)")

        st.write("[View the source code](https://github.com/steamship-packages/langchain-production-starter)")
        # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

        if instance:=st.session_state.get("instance"):
            st.write(f"[Manage on Steamship](https://steamship.com/workspaces/{instance.handle}/packages/{instance.handle})")


def get_api_key():
    if not st.session_state.steamship_api_key:
        st.info("Please add your Steamship API key to continue.")
        st.stop()
    return st.session_state.steamship_api_key


def get_instance() -> PackageInstance:
    instance = st.session_state.get("instance")
    if not instance:
        st.warning("First create your chatbot by clicking \"Chatbot\"")
        st.stop()
    else:
        return instance