# US Stock Market Analysis: Crash & News Correlation Tool

A powerful analytics engine designed to detect, categorize, and explain US stock market crashes across the **NASDAQ**, **S&P 500**, and **Dow Jones Industrial Average (DOW)**.

## 📊 Overview

This project provides an end-to-end pipeline for market volatility research. It goes beyond simple price tracking by automatically identifying "Crash Clusters" and correlating them with global news events to determine the probable cause of investor panic.

## 🚀 Key Features

- **Automated Crash Detection**: 
    - **Single-Day**: Detects price drops of 3% or more within a single trading session.
    - **Multi-Day**: Identifies multi-session downward trends and calculates total cumulative loss.
- **AI-Driven Cause Analysis**: 
    - Automatically scrapes vast amounts of financial news using specialized scrapers.
    - Utilizes **FinBERT (Financial BERT)** to perform deep sentiment analysis and contextual understanding of the news.
    - Summarizes complex market events into a concise **"Crash Summary"**.
    - Identifies and highlights the **"Main Cause"** for every major volatility event.
- **Multi-Index Correlation**: Syncs data across the three major US indexes to find systemic "All-Index" market meltdowns.
- **Interactive Dashboard**: A Flask-based web interface for visualizing crash history, severity, and recovery timelines.
- **Comprehensive Reporting**: Exports detailed analysis to CSV and SQLite for further quantitative research.

## 🛠️ Tech Stack

- **Backend**: Python (Flask)
- **Data Science**: Pandas, NumPy
- **AI & NLP**: **FinBERT** (for financial sentiment and cause analysis)
- **Database**: SQLite3
- **APIs**: NewsData.io
- **Frontend**: HTML5, CSS3, JavaScript (D3.js or Chart.js for visualization)

## 📂 Project Structure

```text
├── Backend6App.py          # Flask API and Web Server
├── Stockmarket5.py         # Detection Logic & Data Processing
├── CauseAnalysis6.py       # Sentiment & Root Cause Analysis
├── ScrapingLast1_Stocks.db # SQLite Database of processed events
├── market5_crash_report_1.csv # Generated analytical report
└── templates/              # Dashboard UI files
```

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9+
- NewsData.io API Key

### 2. Installation
```bash
# Install dependencies
pip install flask flask-cors pandas requests
```

### 3. Configuration
Ensure your index history data (JSON format) is placed in the designated input directory specified in `Stockmarket5.py`.

## 🏃 Usage

1. **Process Market Data**:
   Run the analysis script to detect crashes and pull news data.
   ```bash
   python Stockmarket5.py
   ```

2. **Launch the Dashboard**:
   Start the Flask server to view the results interactively.
   ```bash
   python Backend6App.py
   ```
   Open `http://localhost:5000` in your browser.

## 📈 Methodology
- **Crash Threshold**: Defaulted to 3% drop from Open to Close.
- **Multi-Day Logic**: Tracks consecutive "Lower Lows" to define a single crash event window.
- **Recovery Tracking**: Calculates the number of days taken for the index to return to pre-crash levels.

## 🔒 Licenses & Disclaimer
*This tool is for educational and research purposes only. It does not constituent financial advice.*

<!-- gitpulse:contribution index="1" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="2" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="3" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="4" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="5" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="6" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="7" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="8" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="9" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="10" timestamp="2026-05-16" -->
<!-- gitpulse:contribution index="11" timestamp="2026-05-20" -->