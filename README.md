# 🤖 AI-Powered Stock Advisor

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange?style=for-the-badge)](https://www.langchain.com/)

An intelligent, full-stack application that provides sophisticated stock market analysis using a multi-tool AI system. This project leverages Generative AI to deliver both broad market insights and deep fundamental analysis for individual stocks, all within a beautiful and responsive Streamlit interface.

---

### ✨ Key Features

The application is a powerful financial tool with two distinct modes, accessible from the sidebar:

#### **📈 Market Analysis**
- **Automated Stock Screening:** Scans the market using custom strategies for **'Smart Money'** and **'Momentum'** to identify top-performing stocks.
- **Real-Time News Sentiment:** Fetches and analyzes the latest news for top stocks to gauge market sentiment.
- **AI-Powered Summary:** Generates a detailed, structured report including an **Executive Summary, Top Picks, Risk Assessment, and an Actionable Plan**, tailored to your selected risk profile.
- **Data Visualization:** Includes interactive bubble charts to visualize the performance and RSI of screened stocks.

#### **🔍 Fundamental Analysis**
- **Deep-Dive Analysis:** Select a specific stock (e.g., `RELIANCE.NS`) for a comprehensive fundamental review.
- **Key Financial Metrics:** Fetches and displays critical data like P/E Ratio, P/B Ratio, ROE, and Market Cap.
- **AI-Generated Report:** Provides a qualitative summary of the company's financial health, valuation, and investment outlook.
- **Interactive Price Charts:** Displays a detailed 2-year price history with 50-day and 200-day moving averages.

#### **🎨 User Interface**
- **Modern UI:** A clean, intuitive interface with a beautiful light theme as the default.
- **Dark Mode:** A convenient toggle for users who prefer a dark theme.
- **Performance Optimized:** Uses caching and lazy loading with expanders to ensure a fast and responsive user experience.

---

### 🛠️ Tech Stack & Architecture

The application is built with a decoupled architecture, separating the Streamlit frontend from the Python backend logic. This ensures performance and maintainability.

- **Frontend:** Streamlit
- **Backend & Data Manipulation:** Python, Pandas, NumPy
- **AI & LLM Integration:** LangChain, `langchain-openai`
- **Data Sources:** `yfinance`, `requests`, `beautifulsoup4`, `duckduckgo-search`, `newspaper3k`
- **Plotting:** Matplotlib, Seaborn, Plotly

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/shettysushanth933/stock-advisor>
   cd <your-folder>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up API keys**
   Create a `.streamlit/secrets.toml` file:
   ```toml
   TOGETHER_API_KEY = "your_together_ai_api_key_here"
   ```

## 🎯 Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the app**
   Open your browser and go to `http://localhost:8501`

3. **Start analyzing**
   - Enter queries like "Find top momentum stocks"
   - Adjust risk tolerance and analysis depth
   - View comprehensive analysis results

## 📊 Main Components

### `app.py` - Main Application
- Streamlit frontend with chat interface
- LangGraph agent orchestration
- Real-time stock analysis and visualization

### `stock_analysis.py` - Analysis Modules
- Technical indicators and screening logic
- Data processing and visualization functions

### `requirements.txt` - Dependencies
- All required Python packages and versions

## 🔧 Configuration

### API Keys Required
- **Together AI API Key**: For LLM-powered analysis
- Set in `.streamlit/secrets.toml` or environment variables

### Customization Options
- Risk tolerance levels (Conservative, Moderate, Aggressive)
- Analysis depth (Quick, Standard, Deep)
- Theme preferences (Dark/Light mode)

## 📈 Features in Detail

### Stock Screening
- **Smart Money Strategy**: EMA crossovers with RSI and MACD
- **Momentum Strategy**: High RSI stocks above 200 EMA
- **Cross-Strategy Confirmation**: Stocks appearing in multiple strategies

### News Analysis
- Real-time news fetching for top stocks
- AI-powered sentiment analysis
- Investment action recommendations

### Visualization
- Interactive bubble charts
- Risk/Reward ratio calculations
- Performance metrics display

## 🚨 Disclaimer

⚠️ **Important**: This application is for educational and informational purposes only. All analysis and recommendations should not be considered as financial advice. Always conduct your own research before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 

---

**Built with ❤️ using Streamlit, LangGraph, and Together AI** 