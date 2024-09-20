import os
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.llms import llamacpp
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import FastAPI
from pydantic import BaseModel


# load embedding model
embed_model_id = "Alibaba-NLP/gte-large-en-v1.5"
device='cuda'
embed_model = HuggingFaceBgeEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device':device,'trust_remote_code':True},
    encode_kwargs={'device':device}
)

# download llama2 7b
os.system("wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf?download=true -O llama-2-7b-chat.Q8_0.gguf")

# initialize llm model using llama_cpp
llm = llamacpp(
    model_path="./llama-2-7b-chat.Q8_0.gguf",
    n_gpu_layers=-1,
    n_ctx=4000,
    temperature=0.1,
    verbose=False,
)

# api initialize fastapi app and variables
app = FastAPI()

web_page_link_old = ""
history_aware_retriever = None
question_answer_chain = None
rag_chain = None
store = {}
conversational_rag_chain = None

# input data schema validation using pydantic basemodel

class Query(BaseModel):
    input: str
    session_id: str
    web_page_link: str


# function to index webpage and make chat_history based RAG Chatbot
def initialize_components(query):
    global web_page_link_old,vectorstore,history_aware_retriever,question_answer_chain,rag_chain, conversational_rag_chain,store

    web_page_link = query.web_page_link

    if web_page_link != web_page_link_old:
        print("Initializing components for new web page link:",web_page_link)
        web_page_link_old = web_page_link

        # initiliaze vectorstore
        loader = WebBaseLoader(web_page_link_old)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,  chunk_overlap=100)
        all_splits =  text_splitter.split_documents(data)

        try: 
            vectorstore.delete_collection()
            del vectorstore
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
        except:
            vectorstore = Chroma.from_documents(documents=all_splits,embedding=embed_model)

            # Initialize retriver
            retriever = vectorstore.as_retriever(search_kwargs={"k":2})

            # Initialize  history_aware_retriever
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_message(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )

            history_aware_retriever = create_history_aware_retriever(
                llm,retriever,contextualize_q_prompt
            )

            # Initialize question_answer_chain
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Strictly use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise and short."
                "\n\n"
                "{context}"
            )

            qa_prompt  = ChatPromptTemplate.from_messages(
                [
                    ("system",system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )

            question_answer_chain  = create_stuff_documents_chain(llm, qa_prompt)

            # Initialize rag_chain which uses history_aware_retriver to  answer the question
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Initialize store
            store = {}

            def get_session_history(session_id: str)  -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id]  = ChatMessageHistory()
                return store[session_id]
            # Initialize conversational_rag_chain
            conversational_rag_chain  =  RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            print("Using existing components for web page link:",web_page_link)

# function to run input
def process_query(query):
    if query.web_page_link  ==  "":
        return  {"answer": "please put webpage link"}
    
    initialize_components(query)

    response = conversational_rag_chain.invoke(
        {"input":query.input},
        config= {
            "configurable": {"session_id": query.sesson_id}
        }
    )

# POST METHOD
@app.post("/qury")
def query(query: Query):
    return process_query(query)

# run api using uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,port=8000)