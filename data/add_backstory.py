from pathlib import Path

import click
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from steamship import Steamship
from steamship_langchain.vectorstores import SteamshipVectorStore


@click.command()
def add_backstory():
    companion_name = click.prompt("What's the name of your companion?", type=str)
    workspace = click.prompt("What's the name of your workspace", type=str)
    p = (Path(__file__) / ".." / companion_name).resolve()
    client = Steamship(workspace=workspace)
    if not p.exists():
        print(f"Couldn't find data for {companion_name}. Consider adding txt files to {p.resolve()}")
    else:
        for document in p.iterdir():
            d = Document(page_content=document.open().read(), metadata={"source": p.name})

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=300
            ).split_documents([d])

            vector_store = SteamshipVectorStore(
                client=client,
                embedding="text-embedding-ada-002",
                index_name=companion_name.lower(),
            )
            vector_store.index.reset()
            vector_store.add_documents(chunks)


if __name__ == '__main__':
    add_backstory()
