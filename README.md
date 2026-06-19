# 📊 Pro Trading Intelligence Dashboard

A professional-grade **portfolio analytics and investment intelligence platform** built using Python, Streamlit, and AI.

This system goes beyond basic portfolio tracking by integrating:
- Real-time market data
- Institutional-level risk analytics
- Portfolio performance benchmarking
- AI-powered investment insights using local LLMs

---

## 🚀 Key Features

### 📈 Portfolio Analytics
- Real-time portfolio valuation using live market data (Yahoo Finance)
- Unrealized Profit & Loss (PnL) tracking
- Portfolio weight distribution
- Sector-level breakdown

### 📊 Performance & Benchmarking
- Portfolio performance curve (cumulative returns)
- Benchmark comparison (NIFTY 50 index)
- Normalized return analysis

### ⚠️ Risk Management Engine
- Sharpe Ratio (risk-adjusted returns)
- CAPM Beta (market sensitivity)
- Rolling Beta trends
- Concentration risk detection
- Volatility estimation

### 🧠 AI-Powered Insights
- Portfolio analysis using local LLM (Ollama)
- Natural language investment explanations
- Institutional-style commentary generation

### 🎛️ Interactive Dashboard
- Sector filtering
- Stock-level filtering
- Exposure-based filters (PnL, weight)
- Dynamic portfolio slicing

---

## 🧠 Tech Stack

- Python 3
- Streamlit
- Pandas & NumPy
- Plotly (interactive charts)
- Yahoo Finance API (`yfinance`)
- Scikit-learn (CAPM regression)
- Ollama (Local LLM - Qwen)

---

## 🏗️ Architecture

```
CSV Portfolio → Data Loader → Live Market Prices → 
Portfolio Engine → Risk Analytics Layer → 
Visualization Dashboard → AI Insights Layer
```

---

## 📂 Project Structure

```
AI_Portfolio_Manager/
│
├── app.py                  # Main Streamlit dashboard
├── llm.py                  # AI insights engine (Ollama)
├── portfolio.py            # Portfolio utilities
├── main.py                 # (optional runner script)
│
├── data/
│   └── portfolio.csv       # User holdings
│
├── auth/
│   └── users.yaml          # Authentication config
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Example Metrics Produced

- Portfolio Value (₹)
- Annualized Returns
- Sharpe Ratio (Risk-adjusted return)
- Beta vs NIFTY
- Sector exposure
- Stock-level contribution to returns

---

## 🧪 AI Insight Example

> "The portfolio shows moderate exposure to high-beta IT stocks, leading to elevated volatility compared to the NIFTY benchmark. Risk-adjusted returns are stable, but concentration risk is present in top holdings. Consider diversification into defensive sectors."

---

## 📦 Installation & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements (Roadmap)

- Portfolio optimization engine (Markowitz model)
- Monte Carlo simulation for risk forecasting
- Real-time trading simulation engine
- Multi-user SaaS architecture
- Live broker integration (paper trading)
- Advanced factor models (Fama-French)

---

## 📌 Project Summary

This project demonstrates a **real-world fintech-style analytics system** combining:

- Financial data engineering
- Portfolio risk modeling
- Machine learning regression (CAPM beta)
- AI-driven investment commentary
- Interactive dashboard development

It is designed as a **Bloomberg-style prototype dashboard** for portfolio intelligence and decision support.

---

## 👤 Author

Built by Hardik Sharma  
GitHub: https://github.com/hardiksharma0993