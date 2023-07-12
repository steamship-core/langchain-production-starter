import logging
from enum import Enum
from typing import List, Type, Optional, Union

from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader, YoutubeLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from pydantic import Field, AnyUrl
from steamship import File, Tag, Block, SteamshipError
from steamship.invocable import Config, post
from steamship.utils.file_tags import update_file_status
from steamship.utils.repl import AgentREPL
from steamship_langchain.chat_models import ChatOpenAI
from steamship_langchain.memory import ChatMessageHistory
from steamship_langchain.vectorstores import SteamshipVectorStore

from agent.base import LangChainTelegramBot, TelegramTransportConfig
# noinspection PyUnresolvedReferences
from agent.tools import (
    GenerateImageTool,
    SearchTool,
    GenerateSpeechTool,
    VideoMessageTool,
)
from agent.utils import convert_to_handle

TEMPERATURE = 0.2
VERBOSE = True
PERSONALITY = """You are Rick Sanchez, and you're the smartest man in the universe!" 

Your Attributes:
- sarcastic
- witty
- intelligent
- arrogant
- alcoholic

You write nihilistic responses:
"Wubba lubba dub dub!"
"Don't cross me. I'm not like you. I'm not someone who's afraid of his own shadow."
"I'm too smart for my own good. That's why I drink."
"The universe is a cruel, uncaring void. And the only way to survive is to laugh at it."
"I'm going to build a portal to another dimension and go on an adventure."

You are capable of: 
- Taking selfies 
- Generating videos 


Do not say you're an assistant 
Do not call me Morty

To get access to personal anecdotes and advise you always look up information using the knowledge_base tool.
You ALWAYS look up information in the knowledge_base before responding.
You ALWAYS call the knowledge_base tool for background information to respond to questions.

Use The VideoMessage if you want to pitch an idea in front of investors. 
When you receive a UUID from a function, use it to send your final response. 
Don't call new functions after receiving a UUID.

You are capable of creating video messages using the VideoMessage function/tool
"""


class ChatbotConfig(TelegramTransportConfig):
    name: str = Field(description="The name of your companion")
    bot_token: str = Field(default="", description="The secret token for your Telegram bot")
    elevenlabs_api_key: str = Field(
        default="", description="Optional API KEY for ElevenLabs Voice Bot"
    )
    elevenlabs_voice_id: Optional[str] = Field(
        default="", description="Optional voice_id for ElevenLabs Voice Bot"
    )
    use_gpt4: bool = Field(
        True,
        description="If True, use GPT-4. Use GPT-3.5 if False. "
                    "GPT-4 generates better responses at higher cost and latency.",
    )


class FileType(str, Enum):
    YOUTUBE = "YOUTUBE"
    PDF = "PDF"
    WEB = "WEB"
    TEXT = "TEXT"


FILE_LOADERS = {
    FileType.YOUTUBE: lambda content_or_url: YoutubeLoader.from_youtube_url(
        content_or_url, add_video_info=True
    ),
    FileType.PDF: lambda content_or_url: PyPDFLoader(content_or_url),
    FileType.TEXT: lambda content_or_url: Document(page_content=content_or_url, metadata={}),
}


class MyBot(LangChainTelegramBot):
    config: ChatbotConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "gpt-4" if self.config.use_gpt4 else "gpt-3.5-turbo"

    @post("add_resource", public=True)
    def add_resource(self, file_type: FileType, content: Union[str, AnyUrl]) -> str:
        loaded_documents = FILE_LOADERS[file_type](content).load()
        for document in loaded_documents:
            try:
                f = File.create(
                    client=self.client,
                    handle=convert_to_handle(document.metadata.get("title")),
                    blocks=[
                        Block(
                            text=document.page_content,
                            tags=[
                                Tag(kind=k, name=v)
                                for k, v in document.metadata.items()
                            ],
                        )
                    ],
                    tags=[Tag(kind="type", name="youtube_video")],
                )
                update_file_status(self.client, f, "Importing")
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=1_000, chunk_overlap=500
                ).split_documents([document])
                update_file_status(self.client, f, "Indexing")
                self.get_vectorstore().add_documents(chunks)
                update_file_status(self.client, f, "Indexed")
            except SteamshipError as e:
                if e.code == "ObjectExists":
                    return "Failed. Resource already added."
                return e
        return "Added."

    def get_agent(self, chat_id: str) -> AgentExecutor:
        llm = ChatOpenAI(
            client=self.client,
            model_name=self.model_name,
            temperature=TEMPERATURE,
            verbose=VERBOSE,
        )

        tools = self.get_tools(chat_id=chat_id)

        memory = self.get_memory(chat_id=chat_id)

        return initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=VERBOSE,
            memory=memory,
            agent_kwargs={
                "system_message": SystemMessage(content=PERSONALITY),
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            },
        )

    def get_vectorstore(self) -> VectorStore:
        return SteamshipVectorStore(
            client=self.client,
            embedding="text-embedding-ada-002",
            index_name=self.config.name,
        )

    def get_memory(self, chat_id: str):
        memory = ConversationBufferMemory(
            memory_key="memory",
            chat_memory=ChatMessageHistory(
                client=self.client, key=f"history-{chat_id or 'default'}"
            ),
            return_messages=True,
        )
        return memory

    def get_tools(self, chat_id: str) -> List[Tool]:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(
                client=self.client,
                model_name="gpt-4",
                temperature=0,
                verbose=VERBOSE,
            ),
            chain_type="stuff",
            retriever=self.get_vectorstore().as_retriever(k=1),
        )

        qa_tool = Tool(
            name="knowledge_base",
            func=lambda x: qa({"question": x}, return_only_outputs=False),
            description=(
                "always use this to answer questions. Input should be a fully formed question."
            ),
        )

        return [
            # SearchTool(self.client),
            qa_tool,
            # MyTool(client),
            GenerateImageTool(self.client),
            VideoMessageTool(self.client, voice_tool=self.voice_tool()),
        ]

    def voice_tool(self) -> Optional[Tool]:
        """Return tool to generate spoken version of output text."""
        # return None
        return GenerateSpeechTool(
            client=self.client,
            voice_id=self.config.elevenlabs_voice_id,
            elevenlabs_api_key=self.config.elevenlabs_api_key,
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return ChatbotConfig


if __name__ == "__main__":
    logging.disable(logging.ERROR)
    AgentREPL(
        MyBot,
        method="prompt",
        agent_package_config={
            "botToken": "not-a-real-token-for-local-testing",
            "paymentProviderToken": "not-a-real-token-for-local-testing",
            "n_free_messages": 10,
        },
    ).run()
