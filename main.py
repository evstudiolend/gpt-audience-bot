from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import uvicorn
import os

app = FastAPI()

class InputData(BaseModel):
    question: str

# Загружаем ключ API из переменной окружения
openai_api_key = os.getenv("OPENAI_API_KEY")

# Создаём LLMChain
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Ты — эксперт по маркетингу. Проанализируй целевую аудиторию для следующего продукта или услуги:

{question}

Проанализируй по следующим пунктам:
1. Сегментация по методике 5W (Who, What, Why, When, Where)
2. Сегментация по признакам:
   • География (регион, климат, плотность населения и т.д.)
   • Социально-демографическая (пол, возраст, доход, образование, профессия)
   • Поведенческая (периодичность, мотивация, поводы, этап клиентского пути)
   • Психографическая (ценности, образ жизни, реакция на инновации)
   • B2B-сегментация (если применимо): отрасль, объём закупок, сезонность, роли в закупочном комитете и пр.
3. Описание болей, страхов, барьеров, альтернатив и критериев выбора
4. Сформулируй 1–2 портрета (персоны) представителей аудитории
5. Итоги — как лучше коммуницировать с этой ЦА

Ответ подай развёрнуто, логично и структурно, как в маркетинговом исследовании. Не придумывай лишнего, а глубоко анализируй.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

@app.post("/analyze")
async def analyze(data: InputData):
    response = chain.run(data.question)
    return {"result": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
