# GPT Audience Bot

Бот, который анализирует целевую аудиторию по авторской методике, используя GPT-4 и LangChain.

## 📦 Установка

1. Создайте `.env` файл и вставьте ваш OpenAI API ключ:

```
OPENAI_API_KEY=your_key_here
```

2. Установите зависимости:

```
pip install -r requirements.txt
```

3. Запустите сервер:

```
uvicorn main:app --reload
```

## 📡 Endpoint

POST `/analyze`

```json
{
  "question": "Опиши целевую аудиторию для онлайн-курса по маркетингу для предпринимателей"
}
```