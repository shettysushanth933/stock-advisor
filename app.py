# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain Imports
from langchain_openai import ChatOpenAI

# Web Scraping & News
from newspaper import Article
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup as bs

# ==============================================================================
# --- LLM & API CONFIGURATION ---
# ==============================================================================

try:
    together_api_key = st.secrets["TOGETHER_API_KEY"]
    llm = ChatOpenAI(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.0,
        openai_api_key=together_api_key,
        openai_api_base="https://api.together.xyz/v1",
        max_tokens=2048
    )
except Exception as e:
    st.error(f"Failed to initialize the Language Model. Please check your API key in secrets.toml. Error: {e}")
    llm = None

# ==============================================================================
# --- DATA FETCHING & ANALYSIS FUNCTIONS (WITH CACHING) ---
# ==============================================================================

@st.cache_data(ttl="15m")
def get_market_scan_data():
    """Performs the Chartink market scan and returns dataframes."""
    print("Fetching market scan data...")
    strategies = {
        'Uptrend Screener': ('https://chartink.com/screener/strategy-1-499', {'scan_clause': '( {cash} ( latest close > latest ema( latest close , 200 ) and latest close > latest ema( latest close , 50 ) and latest ema( latest close , 50 ) > latest ema( latest close , 200 ) and latest rsi( 14 ) > 50 and latest macd signal( 26 , 12 , 9 ) > 0 and latest macd line( 26 , 12 , 9 ) > 0 and latest macd line( 26 , 12 , 9 ) > latest macd signal( 26 , 12 , 9 ) ) )'}),
        'Momentum': ('https://chartink.com/screener/momentum', {'scan_clause': '( {cash} ( latest close > latest ema( latest close , 200 ) and latest rsi( 14 ) > 70 ) )'})
    }
    all_stock_data = {}
    for name, (url, clause) in strategies.items():
        try:
            with requests.Session() as s:
                r = s.get(url)
                soup = bs(r.text, "html.parser")
                csrf = soup.select_one("[name='csrf-token']")['content']
                s.headers['x-csrf-token'] = csrf
                r = s.post('https://chartink.com/screener/process', data=clause)
                df = pd.DataFrame(r.json()['data'])
                if not df.empty:
                    all_stock_data[name] = df
        except Exception as e:
            print(f"Failed to fetch {name} data: {e}")
    return all_stock_data


@st.cache_data(ttl="15m")
def get_fundamental_data(ticker: str):
    """Fetches and caches comprehensive fundamental data for a given stock ticker."""
    print(f"Fetching fundamentals for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            "ticker": ticker, "company_name": info.get('longName'), "sector": info.get('sector'),
            "industry": info.get('industry'), "summary": info.get('longBusinessSummary'), "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'), "pb_ratio": info.get('priceToBook'),
            "dividend_yield": info.get('dividendYield', 0) * 100, "roe": info.get('returnOnEquity'),
            "debt_to_equity": info.get('debtToEquity'),
        }
        return fundamentals
    except Exception as e:
        return {"error": f"Could not retrieve data for {ticker}. It may be an invalid symbol. Error: {e}"}

def run_full_market_analysis(risk_profile: str, analysis_type: str):
    """Orchestrates the entire market analysis workflow, including fetching fundamentals for top picks."""
    screener_data = get_market_scan_data()
    if not screener_data: return {"error": "Could not retrieve any stocks from the market screeners."}
    
    main_df = next(iter(screener_data.values()), pd.DataFrame())
    top_stocks = main_df.head(3)['nsecode'].tolist()
    
    # --- NEW: Fetch fundamentals for the top 3 stocks ---
    top_picks_fundamentals = []
    for symbol in top_stocks:
        # Append .NS for Indian stocks if not present, for yfinance
        ticker = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
        f_data = get_fundamental_data(ticker)
        if "error" not in f_data:
            top_picks_fundamentals.append(f_data)

    news_articles, sentiments = [], []
    for symbol in top_stocks:
        try:
            news_results = DDGS().news(f"{symbol} stock news", max_results=1)
            if news_results:
                article_info = news_results[0]
                article = Article(article_info['url']); article.download(); article.parse()
                news_articles.append({"symbol": symbol, "title": article.title, "url": article_info['url'], "content": article.text})
        except Exception as e: print(f"News error for {symbol}: {e}")

    if llm:
        for article in news_articles:
            prompt = f"Analyze the sentiment of this article about {article['symbol']} as Positive, Neutral, or Negative, with a brief reason (under 15 words). Format as: Sentiment: [sentiment], Reason: [reason]"
            sentiments.append({**article, "sentiment_analysis": llm.invoke(prompt).content})
            
    llm_summary = "AI summary could not be generated."
    if llm:
        stock_info = main_df.head(5)[['nsecode', 'close', 'per_chg']].to_string(index=False)
        sentiment_summary = "\n".join([f"- {s['symbol']}: {s['sentiment_analysis']}" for s in sentiments])
        
        # --- NEW: Updated prompt to ask for beginner insights ---
        summary_prompt = f"""
        You are a financial analyst providing a recommendation for a {risk_profile} investor seeking {analysis_type} depth. Based on the data below, provide a response with markdown headings for '### Executive Summary', '### Top Picks', '### Risk Assessment', and '### Actionable Plan'.

        **For the 'Top Picks' section, for each stock you recommend, you MUST include a one-line 'Beginner's Insight' explaining its key strength in simple, easy-to-understand terms.**

        **Top Stocks Found:**
        {stock_info}

        **Recent News Sentiment:**
        {sentiment_summary}
        """
        llm_summary = llm.invoke(summary_prompt).content
        
    return {
        "screener_data": screener_data, 
        "news_sentiments": sentiments, 
        "llm_summary": llm_summary,
        "top_picks_fundamentals": top_picks_fundamentals # <-- NEW: Add fundamentals to the result
    }


def generate_fundamental_summary(data: dict):
    """Generates an LLM summary for the provided fundamental data."""
    def format_metric(value, format_str="{:.2f}"):
        if value is None or not isinstance(value, (int, float)): return "N/A"
        return format_str.format(value)
    prompt = f"You are a senior financial analyst. Based on the following data for {data.get('company_name', data.get('ticker'))}, provide a concise summary of its financial health.\n\n**Key Metrics:**\n- P/E Ratio: {format_metric(data.get('pe_ratio'))}\n- P/B Ratio: {format_metric(data.get('pb_ratio'))}\n- ROE: {format_metric(data.get('roe'), format_str='{:.2%}')}\n- Debt-to-Equity: {format_metric(data.get('debt_to_equity'))}\n\n**Business Summary:**\n{data.get('summary')}\n\n**Analysis Task:**\n1. Briefly overview the company.\n2. Analyze its valuation.\n3. Assess financial stability.\n4. Conclude with a final investment outlook."
    if llm: return llm.invoke(prompt).content
    return "LLM not available for analysis."

# ==============================================================================
# --- PLOTTING FUNCTIONS (WITH CACHING) ---
# ==============================================================================

@st.cache_data
def create_market_scan_plot(df_input: pd.DataFrame, strategy_name: str, is_dark_mode: bool):
    """Generates the bubble chart for the market scan."""
    df = df_input.copy()
    if df.empty: return None
    df['per_chg'] = pd.to_numeric(df['per_chg'], errors='coerce')
    df['rsi'] = pd.to_numeric(df.get('rsi', 0), errors='coerce')
    df['close'] = pd.to_numeric(df.get('close', 0), errors='coerce')
    plt.style.use('dark_background' if is_dark_mode else 'seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='rsi', y='per_chg', size='close', sizes=(50, 500), alpha=0.7, ax=ax)
    title_color = 'white' if is_dark_mode else 'black'
    ax.set_title(f'{strategy_name} - Performance vs. RSI', color=title_color)
    ax.set_xlabel('RSI (14)', color=title_color)
    ax.set_ylabel('Percentage Change', color=title_color)
    if is_dark_mode: ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    plt.tight_layout(); return fig

@st.cache_data
def create_price_history_plot(ticker_str: str, is_dark_mode: bool):
    """Creates a Plotly chart for stock price history."""
    stock = yf.Ticker(ticker_str)
    hist = stock.history(period="2y")
    if hist.empty: return None
    hist['MA_50'] = hist['Close'].rolling(window=50).mean()
    hist['MA_200'] = hist['Close'].rolling(window=200).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_50'], name='50-Day MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_200'], name='200-Day MA'), row=1, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'), row=2, col=1)
    fig.update_layout(title_text=f"{ticker_str} Price History", template="plotly_dark" if is_dark_mode else "plotly_white", height=600, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==============================================================================
# --- STREAMLIT UI ---
# ==============================================================================

# Page Config & Theme
st.set_page_config(page_title="AI Stock Advisor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Beautiful UI & Theme Management ---
def get_theme_colors():
    if st.session_state.dark_mode:
        return {"primary_bg": "#1e1e1e", "secondary_bg": "#2d2d2d", "text_primary": "#ffffff", "accent_color": "#667eea", "header_gradient": "linear-gradient(90deg, #2c3e50 0%, #3498db 100%)", "info_bg": "#262730"}
    else:
        return {"primary_bg": "#FFFFFF", "secondary_bg": "#F0F2F6", "text_primary": "#1E1E1E", "accent_color": "#0068C9", "header_gradient": "linear-gradient(90deg, #0068C9 0%, #00BFFF 100%)", "info_bg": "rgba(0, 104, 201, 0.1)"}

def apply_theme_css():
    colors = get_theme_colors()
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {colors['primary_bg']}; }}
        .main-header {{ background: {colors['header_gradient']}; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .main-header h1, .main-header p {{ color: white !important; }}
        section[data-testid="stSidebar"] {{ background-color: {colors['secondary_bg']} !important; }}
        body, .stApp, .main, p, li, .stMarkdown,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], 
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] .st-bq,
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
        [data-testid="stMetric"] label, [data-testid="stMetric"] div,
        [data-testid="stInfo"], [data-testid="stText"],
        .stAlert [data-testid="stMarkdownContainer"] p {{ color: {colors['text_primary']} !important; }}
        .stButton>button {{ border-radius: 20px; border: 1px solid {colors['accent_color']}; background-color: transparent; color: {colors['accent_color']}; transition: all 0.2s; }}
        .stButton>button:hover {{ background-color: {colors['accent_color']}; color: white !important; }}
        [data-testid="stExpander"] {{ border: 1px solid #E0E0E0; border-radius: 8px; }}
        .stInfo {{ background-color: {colors['info_bg']}; }}
    </style>
    """, unsafe_allow_html=True)

# Initialize and apply theme
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False
apply_theme_css()

# Header
st.markdown('<div class="main-header"><h1>🤖 AI-Powered Stock Advisor</h1><p>From broad market scans to deep fundamental analysis</p></div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose your analysis tool", ("Market Analysis", "Fundamental Analysis"))
st.sidebar.markdown("---")
st.sidebar.subheader("UI Settings")
st.sidebar.toggle("Dark Mode", key="dark_mode")

risk_profile, analysis_type = "Moderate", "Standard"
if app_mode == "Market Analysis":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Market Scan Options")
    risk_profile = st.sidebar.select_slider("Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"], value="Moderate")
    analysis_type = st.sidebar.selectbox("Analysis Depth", ["Quick", "Standard", "Deep"], index=1)

# --- Main App Logic ---
if app_mode == "Market Analysis":
    st.header("📈 General Market Analysis")
    st.write("Scan the market for stocks showing strong momentum and smart money inflows, then get a comprehensive AI-powered outlook.")
    
    if st.button("🚀 Run Full Market Scan"):
        with st.spinner("🤖 Performing multi-stage market analysis... (This can take a moment)"):
            st.session_state.analysis_results = run_full_market_analysis(risk_profile, analysis_type)
    
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        if "error" in results:
            st.error(results["error"])
        else:
            st.success("Market analysis complete!")
            st.subheader("💡 AI-Powered Insights")
            st.markdown(results.get("llm_summary", "No summary available."))

            # --- NEW: Display Fundamental Snapshots of Top Picks ---
            top_picks_data = results.get("top_picks_fundamentals", [])
            if top_picks_data:
                st.markdown("---")
                st.subheader("🔍 Fundamental Snapshot of Top Picks")
                for data in top_picks_data:
                    with st.expander(f"**{data.get('company_name', data.get('ticker'))}**"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("P/E Ratio", f"{data.get('pe_ratio'):.2f}" if data.get('pe_ratio') else "N/A")
                        col2.metric("ROE", f"{data.get('roe')*100:.2f}%" if data.get('roe') else "N/A")
                        col3.metric("Debt/Equity", f"{data.get('debt_to_equity'):.2f}" if data.get('debt_to_equity') else "N/A")

            with st.expander("📰 View News Sentiment Analysis"):
                sentiments = results.get("news_sentiments", [])
                if sentiments:
                    for item in sentiments:
                        st.info(f"**[{item.get('symbol', 'N/A')}]**: [{item.get('title', 'N/A')}]({item.get('url', '#')})")
                        st.text(item.get('sentiment_analysis', 'N/A'))
                else: st.write("No news articles were analyzed.")
            
            with st.expander("📊 View Detailed Screener Data & Plots"):
                strategies = results.get("screener_data", {})
                if strategies:
                    for name, df in strategies.items():
                        st.markdown(f"#### Top Stocks for: {name}")
                        st.dataframe(df.head())
                        plot_fig = create_market_scan_plot(df, name, st.session_state.dark_mode)
                        if plot_fig: 
                            st.pyplot(plot_fig)
                            # --- NEW: Add beginner-friendly insight for the chart ---
                            st.caption("This chart shows which stocks have strong recent performance (high on the vertical axis) vs. their momentum score (RSI). Stocks in the top-right are often strong candidates.")
                else: st.write("No screener data to display.")

elif app_mode == "Fundamental Analysis":
    st.header("🔍 Fundamental Stock Analysis")
    
    indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'ITC.NS', 'HINDUNILVR.NS']
    selected_stock = st.selectbox("Select a stock for a deep-dive analysis", indian_stocks)

    if st.button(f"🔬 Analyze {selected_stock}"):
        with st.spinner(f"🤖 Performing fundamental analysis for {selected_stock}..."):
            st.session_state.fundamental_data = get_fundamental_data(selected_stock)
    
    if 'fundamental_data' in st.session_state and st.session_state.fundamental_data.get('ticker') == selected_stock:
        data = st.session_state.fundamental_data
        if "error" in data:
            st.error(data["error"])
        else:
            st.success(f"Analysis for {data.get('company_name')} complete!")
            st.subheader("💡 AI-Powered Summary")
            with st.spinner("Generating AI summary..."):
                llm_summary = generate_fundamental_summary(data)
            st.markdown(llm_summary)
            
            st.markdown("---")
            st.subheader("📊 Key Fundamentals")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Market Cap (Cr)", f"₹{data.get('market_cap', 0) / 1e7:.2f}")
            col2.metric("P/E Ratio", f"{data.get('pe_ratio'):.2f}" if data.get('pe_ratio') else "N/A")
            col3.metric("P/B Ratio", f"{data.get('pb_ratio'):.2f}" if data.get('pb_ratio') else "N/A")
            col4.metric("ROE", f"{data.get('roe')*100:.2f}%" if data.get('roe') else "N/A")

            with st.expander("📈 View Detailed Price Chart"):
                price_plot = create_price_history_plot(selected_stock, st.session_state.dark_mode)
                if price_plot: 
                    st.plotly_chart(price_plot, use_container_width=True)
                    # --- NEW: Add beginner-friendly insight for the chart ---
                    st.caption("This chart shows the stock's price over the last two years. The orange and red lines are the 50-day and 200-day moving averages, which help identify the trend direction.")
                else: st.write("Could not generate price history plot.")