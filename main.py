import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

knowledge_dir = "knowledge"
docs = []
for filename in os.listdir(knowledge_dir):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(knowledge_dir, filename), encoding="utf-8")
        docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(split_docs, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    question = data.get("question", "")
    response = qa_chain.run(question)
    return {"answer": response}