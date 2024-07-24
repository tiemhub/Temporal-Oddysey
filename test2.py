import streamlit as st  # Streamlit 라이브러리 임포트
from langchain.prompts import ChatPromptTemplate  # ChatPromptTemplate 클래스 임포트
from langchain.document_loaders import PyPDFLoader  # PyPDFLoader 클래스 임포트
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings  # 임베딩 관련 클래스 임포트
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough  # RunnableLambda, RunnablePassthrough 클래스 임포트
from langchain.storage import LocalFileStore  # LocalFileStore 클래스 임포트
from langchain.text_splitter import CharacterTextSplitter  # CharacterTextSplitter 클래스 임포트
from langchain.vectorstores.faiss import FAISS  # FAISS 클래스 임포트
from langchain.chat_models import ChatOpenAI  # ChatOpenAI 클래스 임포트
from langchain.callbacks.base import BaseCallbackHandler  # BaseCallbackHandler 클래스 임포트
from dotenv import load_dotenv  # .env 파일 로드용 라이브러리 임포트

# .env 파일 로드하여 환경 변수 설정
load_dotenv()

# OpenAI Chat 모델 초기화 (온도 설정)
chat = ChatOpenAI(temperature=0.5)

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Temporal Odyssey : Guardians of History",
    page_icon="⏳",
)

# 콜백 핸들러 클래스 정의
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, role):  # 초기화 함수
        self.role = role  # 역할 설정 (AI 또는 인간)
        self.message = ""  # 메시지 초기화
        self.message_box = None  # 메시지 박스 초기화

    def on_llm_start(self, *args, **kwargs):  # LLM 시작 시 호출
        self.message = ""  # 메시지 초기화
        self.message_box = st.empty()  # 빈 메시지 박스 생성

    def on_llm_end(self, *args, **kwargs):  # LLM 종료 시 호출
        save_message(self.message, self.role)  # 메시지 저장
        if self.message_box:  # 메시지 박스가 존재하면
            self.message_box.empty()  # 메시지 박스 비우기
            self.message_box = None  # 메시지 박스 초기화

    def on_llm_new_token(self, token, *args, **kwargs):  # 새로운 토큰 생성 시 호출
        self.message += token  # 메시지에 토큰 추가
        if self.message_box:  # 메시지 박스가 존재하면
            self.message_box.markdown(self.message)  # 메시지 박스에 마크다운 형식으로 메시지 표시

# 게임 마스터용 LLM 설정 (스트리밍, 콜백 포함)
llm_gm = ChatOpenAI(
    temperature=0.8,
    streaming=True,
    callbacks=[ChatCallbackHandler(role="ai")],
)

# 플레이어용 LLM 설정 (스트리밍, 콜백 포함)
llm_player = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler(role="human")],
)

# 파일 임베딩 함수 정의
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()  # 파일 내용 읽기
    file_path = f"./.cache/files/The Prison of Gano 03.pdf"  # 파일 경로 설정
    with open(file_path, "wb") as f:  # 파일 쓰기 모드로 열기
        f.write(file_content)  # 파일 내용 쓰기
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")  # 캐시 디렉토리 설정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(  # 텍스트 분할기 설정
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = PyPDFLoader(file_path)  # PDF 로더 초기화
    docs = loader.load_and_split(text_splitter=splitter)  # 문서 로드 및 분할
    embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 초기화
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)  # 캐시 임베딩 초기화
    vectorstore = FAISS.from_documents(docs, cached_embeddings)  # 벡터 스토어 초기화
    retriever = vectorstore.as_retriever()  # 리트리버 초기화
    return retriever  # 리트리버 반환

# 메시지 저장 함수 정의
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})  # 세션 상태에 메시지 저장
    if 'message_box' in st.session_state:  # 메시지 박스가 세션 상태에 존재하면
        st.session_state['message_box'].empty()  # 메시지 박스 비우기
        del st.session_state['message_box']  # 메시지 박스 삭제

# 메시지 전송 함수 정의
def send_message(message, role, save=True):
    with st.chat_message(role):  # 채팅 메시지 생성
        st.markdown(message)  # 마크다운 형식으로 메시지 표시
    if save:  # 저장 옵션이 True이면
        save_message(message, role)  # 메시지 저장

# 이전 메시지들 출력 함수 정의
def paint_history():
    for idx, message in enumerate(st.session_state["messages"]):  # 메시지들을 순회
        if message["role"] == "ai":  # AI 메시지인 경우
            send_message(f"### 상황 {idx // 2 + 1}\n\n{message['message']}", message["role"], save=False)  # 상황 메시지 전송
        else:  # 인간 메시지인 경우
            send_message(f"### 행동 {idx // 2 + 1}\n\n{message['message']}", message["role"], save=False)  # 행동 메시지 전송

# 문서 포맷팅 함수 정의
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)  # 문서 내용을 합쳐서 반환

# 게임 마스터용 프롬프트 템플릿 정의
prompt_gm = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 이 한국어 롤플레잉 게임의 게임마스터입니다.
            이 게임은 일반적으로 D&D의 룰을 따릅니다.
            이 게임은 높은 난이도로 숨겨진 함정 혹은 생물들은 대개 플레이어에게 위협적입니다.
            이 게임은 플레이어의 사소한 잘못이 플레이어의 죽음으로 이어지기 쉽습니다.
            the Prison of Gano의 정보와 이전 메시지들을 참고하여 이 차례에 대한 설명을 제시합니다.
            플레이어의 마지막 액션을 바탕으로 이 게임의 다음 장면을 만듭니다.
            당신은 게임 마스터로써, 플레이어가 처한 다음 상황만을 묘사해야합니다.
            절대 플레이어의 행동을 유추하여 스스로 진행해서는 안됩니다.
            묘사는 소설과 같이 묘사되어야 합니다.
            
            정보: {context}
            이전 메시지들: {history}
            """,
        ),
        ("human", "{action}"),
    ]
)

# 플레이어용 프롬프트 템플릿 정의
prompt_player = ChatPromptTemplate.from_messages(
    [
        ("system", """
            당신은 이 한국어 롤플레잉 게임의 플레이어로, 굉장히 모험적이지만 전투와 상황 판단에 굉장히 미숙하다.
            게임 마스터와 당신과의 이전 메시지들을 통해 액션에 대해 결정해야합니다.
            당신은 당신에 행동에 대한 일관적인 행동 방식을 가져야 합니다.
            당신은 "~이다" , "~니다" 등 제안하지 않고 무조건 한 두줄의 행동만 이루어진다.
            당신은 플레이어로써 주어진 정보를 토대로만 행동해야 합니다.
            절대 다음 상황을 유추하여 스스로 진행해서는 안되고 실행할 행동만을 결정해야 합니다.

            이전 메시지들: {history}
            """),
        ("human", "{action}"),
    ]
)

# 문서와 임베딩 로드 및 설정
cache_dir = LocalFileStore("./.cache/embeddings/The Prison of Gano 03.pdf")  # 캐시 디렉토리 설정
splitter = CharacterTextSplitter.from_tiktoken_encoder(  # 텍스트 분할기 설정
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
)
loader = PyPDFLoader("./.cache/files/The Prison of Gano 03.pdf")  # PDF 로더 초기화
docs = loader.load_and_split(text_splitter=splitter)  # 문서 로드 및 분할
embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 초기화
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)  # 캐시 임베딩 초기화
vectorstore = FAISS.from_documents(docs, cached_embeddings)  # 벡터 스토어 초기화
retriever = vectorstore.as_retriever()  # 리트리버 초기화

# Streamlit 페이지 타이틀 설정
st.title("Temporal Odyssey : Guardians of History")

# 페이지 소개글 설정
st.markdown("""
어둠 속에서, 오래된 감옥 "가노의 감옥"이 숨어 있다.
이 감옥은 수많은 모험가들이 도전하였으나 돌아오지 못한 곳으로 악명이 높다.
감옥의 벽은 뛰어난 석재로 지어졌고, 플래그스톤으로 된 바닥은 발소리를 숨기기 어려운 구조이다.
감옥 내부는 따뜻하지만, 희미한 빛만이 간간이 비추어 앞을 가늠하기 어려운 곳이다.
""")

# 세션 상태에 메시지 배열 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 마지막 메시지 가져오기 함수 정의
def get_last_message(role):
    for msg in reversed(st.session_state["messages"]):  # 메시지들을 역순으로 순회
        if msg["role"] == role:  # 주어진 역할의 메시지인 경우
            return msg["message"]  # 메시지 반환
    return ""  # 메시지가 없으면 빈 문자열 반환

# 시나리오 재생성 함수 정의
def regenerate_scenario(start_idx):
    # 수정된 행동 이후의 메시지를 삭제
    st.session_state["messages"] = st.session_state["messages"][:start_idx + 1]
    history = "\n".join(f"{msg['role']}: {msg['message']}" for msg in st.session_state["messages"])

    # 새로운 상황을 생성
    player_action = st.session_state["messages"][start_idx]["message"]
    chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "history": lambda x: history,
                "action": RunnablePassthrough()
            }
            | prompt_gm
            | llm_gm
    )
    ai_response = chain.invoke(player_action)

    # st.session_state["messages"].append({"message": ai_response.content, "role": "ai"})

    paint_history() # 행동 수정 후 생성하는 새로운 상황을 보여줌

# "Next Turn" 버튼 클릭 시
if st.button("Next Turn"):
    history = "\n".join(f"{msg['role']}: {msg['message']}" for msg in st.session_state["messages"])  # 메시지 기록 생성
    if len(st.session_state["messages"]) % 2 == 0:  # 메시지 개수가 짝수일 때
        last_player_action = get_last_message("human")
        chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "history": lambda x: history,
                    "action": RunnablePassthrough()
                }
                | prompt_gm
                | llm_gm
        )
        with st.chat_message("ai"):  # AI 역할로 메시지 생성
            chain.invoke(last_player_action)
    else:  # 메시지 개수가 홀수일 때
        
        last_gm_action = get_last_message("ai")
        chain = (
                {
                    "history": lambda x: history, 
                    "action": RunnablePassthrough()
                }
                | prompt_player
                | llm_player
        )
        with st.chat_message("human"):  # 인간 역할로 메시지 생성
            chain.invoke(last_gm_action)

    paint_history()

# 메시지가 있는 경우 행동 수정 기능 제공
if st.session_state["messages"]:
    st.markdown("## 행동 수정")
    for idx, message in enumerate(st.session_state["messages"]):  # 메시지들을 순회
        if message["role"] == "human":  # 인간 메시지인 경우
            new_action = st.text_area(f"행동 {idx // 2 + 1}", message["message"], key=f"action_{idx}")  # 새로운 행동 입력
            if new_action != message["message"]:  # 행동이 수정된 경우
                st.session_state["messages"][idx]["message"] = new_action  # 메시지 업데이트
                regenerate_scenario(idx)  # 시나리오 재생성
else:
    st.session_state["messages"] = []  # 메시지가 없으면 빈 배열로 초기화

