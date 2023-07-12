import re
import urllib.request

import streamlit as st
from googleapiclient.discovery import build


def download_html(url):
    with urllib.request.urlopen(url) as response:
        html = response.read().decode()
    return html


def extract_channel_id_and_name(html_content):
    id_pattern = r'<link rel="canonical" href="https://www.youtube.com/channel/(.*?)"'
    title_pattern = r"<title>(.*?) - YouTube</title>"

    id_match = re.findall(id_pattern, html_content)
    title_match = re.findall(title_pattern, html_content)

    channel_id = id_match[0] if id_match else None
    channel_name = title_match[0] if title_match else None
    return channel_id, channel_name


def get_channel_details(channel_url):
    api_key = st.secrets["google_api_key"]
    youtube = build("youtube", "v3", developerKey=api_key)

    html_content = download_html(channel_url)
    channel_id, channel_name = extract_channel_id_and_name(html_content)
    channel_profile_pic = None
    if not channel_id:
        raise Exception("Could not find channel_id.")
    try:
        request = youtube.channels().list(part="snippet", id=channel_id)
        response = request.execute()

        items = response.get("items", [{}])[0]
        snippet = items.get("snippet", {})
        channel_name = snippet.get("title")
        channel_profile_pic = (
            snippet.get("thumbnails", {}).get("default", {}).get("url")
        )
    except Exception:
        pass
    return channel_name, channel_profile_pic
