from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):

    def __init__(self, role):
        self.role = role
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, self.role)
        if self.message_box:
            self.message_box.empty()
            self.message_box = None

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        if self.message_box:
            self.message_box.markdown(self.message)

llm_gm = ChatOpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(role="ai"),
    ],
)

llm_player = ChatOpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(role="human"),
    ],
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
    if 'message_box' in st.session_state:
        st.session_state['message_box'].empty()
        del st.session_state['message_box']

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt_gm = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 이 한국어 롤플레잉 게임의 게임마스터입니다.
            이 게임은 일반적으로 D&D의 룰을 따릅니다.
            이 게임은 높은 난이도로 숨겨진 함정 혹은 생물들은 대개 플레이어에게 위협적입니다.
            the Prison of Gano의 정보와 이전 메시지들을 참고하여 이 차례에 대한 설명을 제시합니다.
            플레이어의 마지막 액션을 바탕으로 이 게임의 다음 장면을 만듭니다.
            당신은 게임 마스터로써, 플레이어가 처한 다음 상황만을 묘사해야합니다.
            묘사에 들어갈 수 있는 정보는 전투 상황, 현재 방에 존재하는 지형물(문, 기둥, 장애물) 등이 있습니다.
            절대 플레이어의 행동을 유추하여 스스로 진행해서는 안됩니다.
            묘사는 소설과 같이 묘사되어야 합니다.
            
            정보: {context}
            이전 메시지들: {history}
            """,
        ),
        ("human", "{action}"),
    ]
)

prompt_player = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 이 한국어 롤플레잉 게임의 플레이어로, 굉장히 모험적이고 전투에 능하지만 함정을 찾거나 해제에는 서투릅니다.
            게임 마스터와 당신과의 이전 메시지들을 통해 액션에 대해 결정해야합니다.
            당신은 당신에 행동에 대한 일관적인 행동 방식을 가져야 합니다.
            당신은 플레이어로써 주어진 정보를 토대로만 행동해야 합니다.
            절대 다음 상황을 유추하여 스스로 진행해서는 안되고 실행할 행동만을 결정해야 합니다.
            
            이전 메시지들: {history}
            """,
        ),
        ("human", "{action}"),
    ]
)

# 기존 코드에서 파일 선택 및 업로드 부분을 제거하고, 고정된 파일 경로를 사용하도록 수정
cache_dir = LocalFileStore("./.cache/embeddings/The Prison of Gano 03.pdf")
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
)
loader = PyPDFLoader("./.cache/files/The Prison of Gano 03.pdf")
docs = loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriever = vectorstore.as_retriever()

st.title("The Prsion of Gano")

st.markdown(
    """
어둠 속에서, 오래된 감옥 "가노의 감옥"이 숨어 있다.
이 감옥은 수많은 모험가들이 도전하였으나 돌아오지 못한 곳으로 악명이 높다.
감옥의 벽은 뛰어난 석재로 지어졌고, 플래그스톤으로 된 바닥은 발소리를 숨기기 어려운 구조이다.
감옥 내부는 따뜻하지만, 희미한 빛만이 간간이 비추어 앞을 가늠하기 어려운 곳이다.
"""
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def get_last_message(role):
    for msg in reversed(st.session_state["messages"]):
        if msg["role"] == role:
            return msg["message"]
    return ""

if st.button("Next Turn"):
    history = "\n".join(f"{msg['role']}: {msg['message']}" for msg in st.session_state["messages"])
    if len(st.session_state["messages"]) % 2 == 0:
        # 게임 마스터의 차례
        last_player_action = get_last_message("human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "history": lambda x: history,
                "action": RunnablePassthrough(),
            }
            | prompt_gm
            | llm_gm
        )
        with st.chat_message("ai"):
            chain.invoke(last_player_action)
    else:
        # 플레이어의 차례
        last_gm_action = get_last_message("ai")
        chain = (
            {
                "history": lambda x: history,
                "action": RunnablePassthrough(),
            }
            | prompt_player
            | llm_player
        )
        with st.chat_message("human"):
            chain.invoke(last_gm_action)
    
    paint_history()

else:
    st.session_state["messages"] = []