from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
import uvicorn
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.studiolend.ru",  # Домен Tilda
        "http://localhost:3000",      # На случай локального теста
        "https://studiolend.ru"       # Без www — на всякий
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class StepRequest(BaseModel):
    question: str
    step: int

class CheckSubRequest(BaseModel):
    user_id: int

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY не задан в переменных окружения")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHANNEL")

# GPT-4o (основная премиум-модель)
llm_gpt4o = ChatOpenAI(model_name="gpt-4o", temperature=0.4, openai_api_key=openai_api_key)

steps = [
    """1. **Общая характеристика продукта и ниши**
- УТП, позиционирование, премиальность, сегмент рынка""",

    """2. **Сегментация по методике 5W (Марк Шеррингтон):**
- Who — кто потребитель (возраст, пол, должность, отрасль, стиль жизни)
- What — что мы предлагаем (описание, характеристики, кастомизация)
- Why — зачем покупают (мотивация, боли, желания)
- When — когда нужен продукт
- Where — где эту аудиторию можно найти (онлайн и офлайн)""",

    """3. **Сегментация аудитории (B2C и B2B):**

🔸 Географическая:
- Регион, климат, численность, тип населённого пункта, экономическая специализация

🔸 Социально-демографическая:
- Пол, возраст, доход, профессия, этап жизни, образование, состав семьи

🔸 Поведенческая:
- Повод к покупке, частота, мотивация, этап клиентского пути, место покупки

🔸 Психографическая:
- Ценности, стиль жизни, тип восприятия, отношение к новизне

🔸 B2B (если применимо):
- Отрасль, объём закупок, сезонность, структура закупки, роли в комитете""",

    """4. **Поведенческие и эмоциональные аспекты:**
- Боли и проблемы
- Сомнения, страхи, возражения
- Альтернативы
- Критерии выбора продукта и продавца
- Ожидания и как их превзойти
- Кто влияет на решение, с кем советуются""",

    """5. **Путь клиента (Customer Journey):**
- Этапы принятия решения
- Что важно на каждом этапе""",

    """6. **Jobs To Be Done:**
- Что клиент хочет 'нанять' продукт сделать
- Используй формат: «Когда я..., я хочу..., чтобы...»""",

    """7. **Типовые персонажи:**
- Имя, возраст, профессия, стиль жизни, мотивация, страхи, как покупает""",

    """8. **Каналы и поведение:**
- Где представители ЦА обычно ищут решение своей задачи в рамках данной ниши: онлайн и офлайн-каналы (соцсети, сайты, консультации, мероприятия и пр.)
- Как они принимают решение, у кого купить: что для них важно (доверие, экспертность, визуальный стиль, отзывы, личные контакты и т.д.)
- Не ограничивайся шаблоном — анализируй поведение ЦА в контексте ниши продукта""",

    """9. **Выводы и рекомендации:**
- Кому продавать и через что
- Как выделиться, как говорить
- Как обойти конкурентов"""
]

# 🎯 Платный маршрут — GPT-4o
@app.post("/analyze-step")
async def analyze_step(data: StepRequest):
    return await generate_analysis(data, llm_gpt4o)

# 🎯 Бесплатный маршрут — GPT-3.5
@app.post("/analyze-step-free")
async def analyze_step_free(data: StepRequest):
    llm_gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4, openai_api_key=openai_api_key)
    return await generate_analysis(data, llm_gpt35)

class LandingStepRequest(BaseModel):
    data: dict
    step: int

landing_steps = [
    "Первый экран: Заголовок, подзаголовок, буллиты, CTA",
    "Проблема → Решение: словами клиента",
    "Как это работает / Преимущества",
    "Социальные доказательства: отзывы, кейсы, факты",
    "Кейсы: истории клиентов",
    "Подробности оффера и бонусы",
    "Финальный экран: CTA + резюме"
]

@app.post("/generate-landing-step")
async def generate_landing_step(request: LandingStepRequest):
    step_index = request.step - 1
    if step_index < 0 or step_index >= len(landing_steps):
        return {"error": "Неверный номер шага"}

    prompt = build_landing_prompt(request.data, request.step, landing_steps[step_index])
    response = llm_gpt4o.predict(prompt)
    return {"step": request.step, "result": response}

def build_landing_prompt(data, step_num, block_title):
    return f"""
Ты — маркетолог и копирайтер с сильным опытом в продажах. Напиши текст для **блока {step_num}: {block_title}** продающего лендинга. Структура AIDA. Язык клиента. Только конкретика, никакой воды.

📌 Условия:
— Уникальные заголовки и подзаголовки с цифрами, выгодами, фактами
— Выделяй главное: УТП, оффер, сильные смыслы
— Не повторяй клише и общие слова («надежно», «быстро», «высокое качество»)
— Примени результаты анализа ЦА и конкурентов

📦 Данные:
Продукт: {data.get("product_name", "")}
Описание: {data.get("product_description", "")}
Компания: {data.get("company_info", "")}
ЦА: {data.get("audience_analysis", "")}
Конкуренты: {data.get("competitors_info", "")}
Преимущества: {data.get("unique_selling_points", "")}
Оффер: {data.get("main_offer", "")}
Бонусы: {data.get("bonuses", "")}
Кейсы: {data.get("case_studies", "")}
Отзывы: {data.get("testimonials", "")}
Цель: {data.get("goal", "")}
Стиль: {data.get("style", "")}

🎯 Напиши текст только для блока {step_num}. Делай текст ярким, логичным, цепляющим.
"""
    
@app.post("/generate-landing")
async def generate_landing(data: dict):
    import openai
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.path.exists("prompt_landing.txt"):
        return {"error": "⚠️ prompt_landing.txt не найден на сервере. Проверь структуру проекта."}

    try:
        with open("prompt_landing.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()

        for key, value in data.items():
            placeholder = f"{{{{{key}}}}}"
            prompt_template = prompt_template.replace(placeholder, str(value or ""))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.7
        )

        return {"result": response.choices[0].message.content}
    except Exception as e:
        return {"error": f"❌ Ошибка при генерации лендинга: {str(e)}"}

# 🔁 Общая логика анализа
async def generate_analysis(data: StepRequest, llm):
    description = data.question
    step_index = data.step - 1

    if step_index < 0 or step_index >= len(steps):
        return {"error": "Неверный номер шага"}

    step_prompt = f"""Ты — маркетинговый аналитик и бренд-стратег.

🌟 Твоя задача — не просто выдать шаблон, а — понять суть продукта и помочь автору взглянуть на него глазами клиента.

🧠 Пиши, как опытный консультант: живым, понятным языком, уверенно. Делай выводы. Не бойся "зайти дальше", если видишь интересный инсайт.

Описание продукта:
{description}

📌 Формат:
- Используй подзаголовки, списки, таблицы
- Делай ясные выводы
- Избегай сухости и академичности
- Представь, что ты пишешь это для клиента, чтобы он мог реально использовать это для продвижения

🚫 Не выдумывай данные — строй выводы логично на основе описания продукта.

### Шаг {data.step}:
{steps[step_index]}
"""

    response = llm.predict(step_prompt)
    return {"step": data.step, "result": response}

@app.post("/check-subscription")
async def check_subscription(data: CheckSubRequest):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChatMember"
    params = {
        "chat_id": TELEGRAM_CHANNEL,
        "user_id": data.user_id
    }

    try:
        response = requests.get(url, params=params).json()
        status = response.get("result", {}).get("status", "")

        if status in ["member", "administrator", "creator"]:
            return {"subscribed": True}
        else:
            return {"subscribed": False}
    except Exception as e:
        return {"subscribed": False, "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
