# Cohere FastAPI Chatbot with pgvector and Structured JSON Output

This project is an AI-powered chatbot built with **Cohere**, **FastAPI**, and **PostgreSQL with pgvector**.  
It provides intelligent responses, supports Retrieval-Augmented Generation (RAG), stores chat history, and integrates external tools such as a weather API.

## Features
- FastAPI backend for chatbot interaction
- Cohere Command-R model for natural conversation
- pgvector in PostgreSQL for semantic search
- Automatic document embedding and storage
- File-change detection with re-embedding
- Chat history storage in PostgreSQL
- Structured JSON responses with short and detailed answers
- Weather integration via OpenWeather API
- CORS enabled for frontend integration

## Requirements
- Python 3.9+
- PostgreSQL with pgvector extension
- Cohere API key
- OpenWeather API key
