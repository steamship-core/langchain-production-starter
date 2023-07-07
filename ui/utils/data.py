import concurrent
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
import scrapetube
import streamlit as st
from steamship import PackageInstance


def add_resource(invocation_url: str, api_key: str, url: str):
    response = requests.post(
        f"{invocation_url}add_resource",
        json={"file_type": "YOUTUBE", "content": url},
        headers={"Authorization": f"bearer {api_key}"},
    )
    return response.text


def index_youtube_channel(
    channel_url: str, offset: Optional[int] = 0, count: Optional[int] = 10
):
    instance: PackageInstance = st.session_state.instance

    videos = scrapetube.get_channel(channel_url=channel_url)

    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for video in itertools.islice(videos, offset, offset + count + 1):
            video_url = f"https://www.youtube.com/watch?v={video['videoId']}"
            futures.append(
                executor.submit(
                    add_resource,
                    instance.invocation_url,
                    instance.client.config.api_key,
                    video_url,
                )
            )

    for ix, future in enumerate(concurrent.futures.as_completed(futures)):
        print(future.result())
