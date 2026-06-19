Overview: The AI Portfolio Manager is a Python-based financial assistant that analyzes stock portfolios using live market data and a local Large Language Model (Qwen via Ollama). It helps users understand their investments in simple language by combining real-time data with AI-generated insights.
Features: Fetch real-time stock prices using Yahoo Finance, Calculate total portfolio value dynamically, Basic risk analysis based on diversification, AI-powered portfolio insights using Qwen (local LLM),Interactive command-line assistant
Tech Stack: Python, Pandas, yFinance, Ollama (Qwen 3 8B), VS Code
How It Works: CSV Portfolio → Python Loader → Live Stock Prices → Portfolio Calculator → AI (Qwen) → Insights
Example Usage: What is my portfolio value?
AI Output: Your portfolio is valued at ₹63,910. It is moderately diversified with exposure to IT stocks like TCS and Infosys...
Project Structure: AI_PORTFOLIO_MANAGER/
│
├── main.py
├── llm.py
├── portfolio.py
├── README.md
│
├── data/
│   └── portfolio.csv
│
└── venv/

Project Summary: This project is a simple AI-powered portfolio manager that combines real-time stock data with a local AI model to generate investment insights.
It demonstrates how Python, financial data APIs, and local LLMs can work together to build an intelligent assistant that explains portfolio performance in simple terms.
This is an early version of the project, with scope for future improvements like dashboards, better risk models, and advanced AI agent capabilities.