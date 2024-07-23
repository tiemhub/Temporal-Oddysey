import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQAChain
from langchain.document_loaders import PDFDocumentLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import VectorStoreRetriever

# Load the dungeon content from the PDF
pdf_path = "/mnt/data/The Prison of Gano 03.pdf"
loader = PDFDocumentLoader(pdf_path)
documents = loader.load()

# Split the document into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(texts, embeddings)

# Create retriever
retriever = VectorStoreRetriever(vector_store=vector_store)

# Initialize Langchain OpenAI models
llm_gm = OpenAI(temperature=0.7)
llm_player = OpenAI(temperature=0.7)

# Create RetrievalQAChain for GM and Player
gm_prompt_template = ChatPromptTemplate.from_template("You are the Game Master for a D&D game. Use the following context to guide the player: {context}. Respond to the player's actions.")
player_prompt_template = ChatPromptTemplate.from_template("You are a player in a D&D game. The Game Master said: {gm_response}. What do you do next?")

gm_qa_chain = RetrievalQAChain(llm=llm_gm, retriever=retriever, prompt=gm_prompt_template)
player_qa_chain = RetrievalQAChain(llm=llm_player, retriever=retriever, prompt=player_prompt_template)

# Streamlit interface
st.title("D&D Chatbot Interaction with RAG")
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Generate GM Turn"):
    if not st.session_state.conversation:
        gm_response = gm_qa_chain.run({"context": dungeon_content, "query": ""})
    else:
        player_response = st.session_state.conversation[-1]["player"]
        gm_response = gm_qa_chain.run({"context": player_response, "query": dungeon_content})
    st.session_state.conversation.append({"gm": gm_response, "player": ""})
    st.write("**Game Master:**", gm_response)

if st.button("Generate Player Turn"):
    if not st.session_state.conversation or not st.session_state.conversation[-1]["gm"]:
        st.warning("Generate GM turn first.")
    else:
        gm_response = st.session_state.conversation[-1]["gm"]
        player_response = player_qa_chain.run({"context": gm_response, "query": ""})
        st.session_state.conversation[-1]["player"] = player_response
        st.write("**Player:**", player_response)

st.write("### Conversation History")
for turn in st.session_state.conversation:
    if turn["gm"]:
        st.write("**Game Master:**", turn["gm"])
    if turn["player"]:
        st.write("**Player:**", turn["player"])
