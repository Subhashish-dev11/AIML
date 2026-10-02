# 🚀 AI Text Summarization App

A simple **AI-powered text summarization web application** built using **FastAPI, Hugging Face Transformers, and T5**.

The application takes user-provided text and generates a concise summary using a Transformer-based NLP model.

---

## 📌 Project Overview

This project demonstrates how a Transformer-based language model can be integrated into a web application for automatic text summarization.

The application contains:

- 🧠 Hugging Face Transformer model
- ⚡ FastAPI backend
- 🌐 HTML/CSS/JavaScript frontend
- 🍎 Local inference using Apple Silicon / CPU
- 📝 Automatic text summarization

---

## 🏗️ Project Architecture

```text
User
  │
  ▼
HTML / CSS / JavaScript
  │
  │ POST /summarize/
  ▼
FastAPI Backend
  │
  ▼
Text Preprocessing
  │
  ▼
T5 / FLAN-T5 Model
  │
  ▼
Generated Summary
  │
  ▼
Web Interface
