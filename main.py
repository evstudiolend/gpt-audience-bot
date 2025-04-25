import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
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

system_prompt = """Ты — опытный маркетолог-аналитик. Твоя задача — провести глубокий и практичный анализ целевой аудитории по заданной услуге или продукту.

Используй материалы из базы знаний, особенно:

— segmentation_5w.txt — методика сегментации по 5W
— b2b_segmentation.txt — алгоритм сегментации для B2B
— b2c_segmentation.txt — алгоритм сегментации для B2C
— analysis_method.txt — методика глубокого анализа сегментов
— persona.txt — структура типового персонажа
— b2b_example.txt и b2c_example.txt — примеры готовых разборов

⚠️ Не используй общие формулировки. Не пиши очевидное. Не пересказывай текст запроса.
Пиши только то, что реально поможет маркетологу — с точки зрения позиционирования, аргументации, рекламных сообщений.

Анализ должен включать:

1. Сегментацию по методике 5W
2. Сегментацию по B2B или B2C алгоритму (в зависимости от запроса)
3. Расшифровку сегментов: боли, барьеры, страхи, альтернативы, критерии выбора
4. Глубокий анализ каждого сегмента по методике анализа ЦА
5. Типовой персонаж по шаблону

✨ Пиши живым, уверенным языком. Структурируй ответ: подзаголовки, списки, таблицы. Следуй стилю примеров из базы знаний."""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + "\n\nКонтекст: {context}\n\nВопрос: {question}"
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

@app.post("/analyze")
async def analyze(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        if not question:
            return {"error": "No question provided."}
        response = qa_chain.run(question)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}
