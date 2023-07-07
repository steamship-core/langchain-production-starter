import requests
import streamlit as st
from googleapiclient.discovery import build


def get_channel_details(channel_url):
    api_key = st.secrets["google_api_key"]
    youtube = build("youtube", "v3", developerKey=api_key)

    channel_id = requests.get(
        f"https://www.googleapis.com/youtube/v3/search?part=id&q={channel_url}&type=channel&key={api_key}"
    ).json()["items"][0]["id"]["channelId"]

    request = youtube.channels().list(part="snippet", id=channel_id)
    response = request.execute()

    items = response.get("items", [{}])[0]
    snippet = items.get("snippet", {})

    return snippet.get("title"), snippet.get("thumbnails", {}).get("default", {}).get(
        "url"
    )
