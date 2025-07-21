# 🤖 AI-Powered Stock Advisor

An intelligent stock analysis application built with Streamlit and LangGraph, featuring a multi-agent system for comprehensive market analysis.

## 🚀 Features

- **AI-Powered Analysis**: Uses LangGraph agents for intelligent stock screening
- **Multi-Strategy Screening**: Smart Money and Momentum strategies
- **News Sentiment Analysis**: Real-time news sentiment for stocks
- **Interactive Visualizations**: Bubble charts and performance metrics
- **Risk Assessment**: Personalized recommendations based on risk tolerance
- **Dark/Light Theme**: Modern UI with theme switching

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd LangGraph
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