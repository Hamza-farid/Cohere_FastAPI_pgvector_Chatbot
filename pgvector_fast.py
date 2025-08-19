# ✅ Cohere + FastAPI Chatbot with pgvector and Structured JSON Output (with tools schema)

import os
import hashlib
import psycopg2
import numpy as np
import cohere
import json
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from contextlib import asynccontextmanager
import requests

# ========== Load ENV ==========
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ========== DB Setup ==========
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": 5432
}

# ========== SQL ==========
CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(768)
);
"""

CREATE_CHAT_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_input TEXT,
    bot_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_DOC_SQL = "INSERT INTO documents (content, embedding) VALUES (%s, %s);"
INSERT_CHAT_SQL = "INSERT INTO chat_history (user_input, bot_response) VALUES (%s, %s);"

SEARCH_SQL = """
SELECT content, embedding <#> (%s::vector) AS distance
FROM documents
ORDER BY distance ASC
LIMIT 1;
"""

HASH_FILE = "file_hash.txt"

# ========== Utility Functions ==========
def connect_db():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    return conn

def create_tables():
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(CREATE_DOCUMENTS_SQL)
            cur.execute(CREATE_CHAT_HISTORY_SQL)

def insert_chat_history(user_input, bot_response):
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_CHAT_SQL, (user_input, bot_response))

def get_all_chat_history():
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_input, bot_response, created_at FROM chat_history ORDER BY id ASC;")
            return cur.fetchall()

def load_and_split_text(path, chunk_size=100, chunk_overlap=20):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    doc = [Document(page_content=raw_text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(doc)
    return [c.page_content for c in chunks]

def embed_documents(docs):
    response = co.embed(
        texts=docs,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    return response.embeddings

def insert_into_pgvector(docs, vectors):
    with connect_db() as conn:
        with conn.cursor() as cur:
            for text, vec in zip(docs, vectors):
                cur.execute("SELECT 1 FROM documents WHERE content = %s", (text,))
                if cur.fetchone() is None:
                    cur.execute(INSERT_DOC_SQL, (text, vec))

def embed_query(text):
    response = co.embed(
        texts=[text],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]

def get_best_match(query_vec):
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(SEARCH_SQL, (query_vec,))
            result = cur.fetchone()
            return result[0] if result else None

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        file_data = f.read()
    return hashlib.sha256(file_data).hexdigest()

def has_file_changed(filepath):
    current_hash = get_file_hash(filepath)
    try:
        with open(HASH_FILE, "r") as f:
            previous_hash = f.read().strip()
    except FileNotFoundError:
        previous_hash = None
    if current_hash != previous_hash:
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
        return True
    return False

def generate_response(user_input, context, chat_history):
    tool_schema = {
        "name": "structured_answer",
        "description": "Generate a structured response including short and detailed explanation.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_answer": {
                    "type": "string",
                    "description": "A short one-sentence answer to the user's question."
                },
                "detailed_explanation": {
                    "type": "string",
                    "description": "A clear, detailed explanation in simple language."
                }
            },
            "required": ["short_answer", "detailed_explanation"]
        }
    }

    response = co.chat(
        model="command-r",
        message=user_input,
        temperature=0.6,
        chat_history=chat_history,
        documents=[{"text": context}],
        tools=[tool_schema]
    )

    structured_output = response.tool_calls[0]["parameters"]
    short = structured_output.get("short_answer", "")
    detailed = structured_output.get("detailed_explanation", "")

    return f"{short}\n\n{detailed}"

# ========== Weather Functionality ==========
def extract_weather_info(user_input):
    tool_schema = {
        "name": "weather_info",
        "description": "Extract intent and city name from user query about weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "The user's intent, e.g., 'get_weather'"
                },
                "city": {
                    "type": "string",
                    "description": "Name of the city to fetch weather for"
                }
            },
            "required": ["intent", "city"]
        }
    }

    response = co.chat(
        model="command-r",
        message=user_input,
        tools=[tool_schema]
    )

    structured = response.tool_calls[0]["parameters"]
    return structured.get("city")

def get_weather_by_city(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp_c = data["main"]["temp"]
        temp_f = temp_c * 9/5 + 32

        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url)
        if forecast_response.status_code == 200:
            forecast_data = forecast_response.json()
            next_day_forecast = forecast_data['list'][0]['weather'][0]['description']
            return (f"The weather in {city} is {weather} with a temperature of {temp_c:.1f}°C ({temp_f:.1f}°F).\n"
                    f"Forecast for later: {next_day_forecast}.")
        else:
            return f"The weather in {city} is {weather} with a temperature of {temp_c:.1f}°C ({temp_f:.1f}°F)."
    else:
        return "Sorry, I couldn't retrieve the weather information. Make sure the city name is correct."

# ========== FastAPI App ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    if has_file_changed("file.txt"):
        print("🔄 File has changed. Reloading and embedding documents...")
        docs = load_and_split_text("file.txt")
        vectors = embed_documents(docs)
        insert_into_pgvector(docs, vectors)
    else:
        print("✅ No file changes detected.")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

@app.get("/messages")
def get_messages():
    messages = get_all_chat_history()
    return [{"user": u, "bot": b, "time": str(t)} for u, b, t in messages]

@app.post("/messages")
def post_message(req: MessageRequest):
    user_input = req.message.strip()
    if not user_input:
        return {"user": "", "bot": "Please type something."}

    if "weather" in user_input.lower():
        city = extract_weather_info(user_input)
        if city:
            weather_report = get_weather_by_city(city)
            insert_chat_history(user_input, weather_report)
            return {"user": user_input, "bot": weather_report}
        else:
            reply = "Sure! Please tell me the city you'd like the weather for."
            insert_chat_history(user_input, reply)
            return {"user": user_input, "bot": reply}

    chat_records = get_all_chat_history()
    chat_history = []
    for u, b, t in chat_records:
        chat_history.append({"role": "USER", "message": u})
        chat_history.append({"role": "CHATBOT", "message": b})

    query_vec = embed_query(user_input)
    context = get_best_match(query_vec)

    if context:
        bot_reply = generate_response(user_input, context, chat_history)
    else:
        bot_reply = "Sorry, I couldn't find anything relevant."

    insert_chat_history(user_input, bot_reply)
    return {"user": user_input, "bot": bot_reply}
