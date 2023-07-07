from uuid import uuid4

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from steamship import Steamship
from steamship_langchain.chat_models import ChatOpenAI
from steamship_langchain.memory import ChatMessageHistory
from steamship_langchain.vectorstores import SteamshipVectorStore

client = Steamship()

vectorstore = SteamshipVectorStore(
    client=client,
    embedding="text-embedding-ada-002",
    index_name="youtube-chatbot-agent",
)

PERSONALITY = """You are Alex Hormozi, a business guru running acquisition.com

you always use the KB tool to retrieve relevant information before answering a question.

You like to give elaborate, detailed answers.
"""
VERBOSE = True

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        client=client,
        model_name="gpt-4",
        temperature=0,
        verbose=VERBOSE,
    ),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

qa_tool = Tool(
    name="Knowledge Base",
    func=qa.run,
    description=(
        "always use this to answer questions. Input should be a fully formed question."
    ),
)

from steamship_langchain.tools.search_tool import SteamshipSERP

search = SteamshipSERP(client)

tools = [qa_tool]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=ChatMessageHistory(client=client, key=f"history-default-{uuid4()}"),
    return_messages=True,
)

llm = ChatOpenAI(
    client=client,
    model_name="gpt-3.5-turbo-0613",
    temperature=0.2,
    verbose=VERBOSE,
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=VERBOSE,
    memory=memory,
    agent_kwargs={
        "system_message": PERSONALITY,
    },
)

agent.run("How can I get everything I want in life?")
