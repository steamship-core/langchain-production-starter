import sys
from pathlib import Path

import streamlit as st
from steamship.cli.create_instance import load_manifest

from utils.youtube import get_channel_details

sys.path.append(str((Path(__file__) / "..").resolve()))
st.set_page_config(page_title="🚢 Youtube to Chatbot")

from utils.data import index_youtube_channel
from utils.utils import get_instance, snake_case
from utils.ux import sidebar, get_api_key

# Start page
st.title("🚢 Youtube-To-Chatbot")
st.write(
    "A project brought to life by [Enias](https://twitter.com/eniascailliau), inspired by [Emmet Halm](https://twitter.com/ehalm_) and [Taranjeet](https://twitter.com/taranjeetio)"
)

sidebar()

manifest = load_manifest()

if not st.session_state.get("instance"):
    channel_url = st.text_input("Youtube URL")
    st.session_state.channel_url = channel_url

    if st.button("🧠 Scrape & Train"):
        try:
            channel_name, channel_thumbnail = get_channel_details(channel_url)
            st.session_state.channel_thumbnail = channel_thumbnail
            st.session_state.channel_name = channel_name = snake_case(channel_name)
        except Exception:
            st.error("Youtube channel not found. Please provide a Youtube channel url")
            st.stop()

        st.session_state.instance = get_instance(channel_name)

        with st.spinner("Loading first 10 youtube video's..."):
            index_youtube_channel(channel_url=channel_url, offset=0, count=10)

        st.balloons()
        st.experimental_rerun()

else:
    channel_name = st.session_state.channel_name
    instance = st.session_state.instance

    st.header(f"Start chatting with Rik")
    st.subheader(
        f"🧠 Rick gleaned wisdom from: {channel_name.replace('_', ' ').title()}"
    )
    st.image(st.session_state.channel_thumbnail)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        get_api_key()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = instance.invoke("prompt", prompt=prompt)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
