import os
import logging

from fastapi import FastAPI, Request
import requests

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/v1/chat/completions"
)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def make_alice_response(body: dict, text: str, end_session: bool = False) -> dict:
    return {
        "version": body.get("version", "1.0"),
        "session": body.get("session", {}),
        "response": {
            "end_session": end_session,
            "text": text,
        },
    }


@app.post("/")
async def main(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {
            "version": "1.0",
            "session": {},
            "response": {
                "end_session": False,
                "text": "Не удалось разобрать запрос.",
            },
        }

    request_data = body.get("request", {})
    user_text = (
        request_data.get("original_utterance")
        or request_data.get("command")
        or ""
    ).strip()

    if not user_text:
        return make_alice_response(
            body,
            "Скажите ваш вопрос.",
            end_session=False,
        )

    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY не задан в переменных окружения")
        return make_alice_response(
            body,
            "Сервис не настроен: отсутствует API-ключ DeepSeek.",
            end_session=False,
        )

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "user", "content": user_text}
                ],
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.exception("Ошибка запроса к DeepSeek: %s", exc)
        return make_alice_response(
            body,
            "Не удалось связаться с нейросетью. Попробуйте позже.",
            end_session=False,
        )

    # Если DeepSeek вернул 4xx/5xx — не пытаемся читать choices
    if not response.ok:
        logger.error(
            "DeepSeek вернул статус %s. Тело ответа: %s",
            response.status_code,
            response.text[:1000],
        )
        return make_alice_response(
            body,
            "Нейросеть временно вернула ошибку. Попробуйте позже.",
            end_session=False,
        )

    try:
        data = response.json()
    except ValueError:
        logger.error("DeepSeek вернул не-JSON ответ: %s", response.text[:1000])
        return make_alice_response(
            body,
            "Нейросеть вернула некорректный ответ. Попробуйте позже.",
            end_session=False,
        )

    choices = data.get("choices") or []
    if not choices:
        logger.error("В ответе DeepSeek нет choices: %s", data)
        return make_alice_response(
            body,
            "Не удалось получить ответ от нейросети. Попробуйте позже.",
            end_session=False,
        )

    message = choices[0].get("message") or {}
    answer = (message.get("content") or "").strip()

    if not answer:
        logger.error("В ответе DeepSeek пустой content: %s", data)
        return make_alice_response(
            body,
            "Нейросеть вернула пустой ответ. Попробуйте переформулировать вопрос.",
            end_session=False,
        )

    return make_alice_response(body, answer, end_session=False)
