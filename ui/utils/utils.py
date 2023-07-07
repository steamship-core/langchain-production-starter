from re import sub
from uuid import uuid1

import streamlit as st
from steamship import Steamship, PackageInstance, SteamshipError
from steamship.cli.create_instance import load_manifest

from utils.ux import get_api_key


def snake_case(s):
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                s.replace('-', ' '))).split()).lower()


def get_instance(channel_name) -> PackageInstance:
    api_key = get_api_key()
    manifest = load_manifest()

    st.text("Creating your own Steamship instance...")

    try:
        client = Steamship(api_key=api_key, workspace=channel_name)
        instance = client.use(manifest.handle, instance_handle=channel_name)
        st.text(
            f"Web URL: https://steamship.com/workspaces/{channel_name}/packages/{instance.handle}"
        )
        st.text(f"API URL: {instance.invocation_url}")
        return instance

    except SteamshipError as e:
        return get_instance(f"{channel_name}-{uuid1()}")
