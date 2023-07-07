import re
import urllib.request

import streamlit as st
from googleapiclient.discovery import build


def download_html(url):
    with urllib.request.urlopen(url) as response:
        html = response.read().decode()
    return html


def extract_channel_id(channel_url):
    html_content = download_html(channel_url)
    pattern = r'<link rel="canonical" href="https://www.youtube.com/channel/(.*?)"'
    matches = re.findall(pattern, html_content)
    return matches[0] if matches else None


def get_channel_details(channel_url):
    api_key = st.secrets["google_api_key"]
    youtube = build("youtube", "v3", developerKey=api_key)

    channel_id = extract_channel_id(channel_url)
    print("channel_id", channel_id)
    if not channel_id:
        raise Exception("Could not find channel_id.")
    request = youtube.channels().list(part="snippet", id=channel_id)
    response = request.execute()

    items = response.get("items", [{}])[0]
    snippet = items.get("snippet", {})

    return snippet.get("title"), snippet.get("thumbnails", {}).get("default", {}).get(
        "url"
    )
