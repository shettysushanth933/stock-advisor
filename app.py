# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pickle
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
from bs4 import BeautifulSoup as bs
from IPython.display import display

# --- LangChain & LangGraph Imports ---
from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from duckduckgo_search import DDGS
from newspaper import Article
from sklearn.cluster import KMeans

# Using st.secrets for API key management
try:
    # For local development, use a secrets.toml file
    # Example secrets.toml:
    # TOGETHER_API_KEY = "your_api_key_here"
    together_api_key = st.secrets["TOGETHER_API_KEY"]
except FileNotFoundError:
    # Fallback for environments without st.secrets (e.g., direct run)
    # Ensure you have set this environment variable
    together_api_key = os.environ.get("TOGETHER_API_KEY", "default_key_if_not_set")

# ==============================================================================
# --- AGENT BACKEND: LANGGRAPH STOCK ANALYSIS AGENT ---
# ==============================================================================

# --- LLM Configuration ---
# # Use a try-except block to handle potential API key errors gracefully
# try:
#     llm = Tongyi(
#         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
#         temperature=0.0,
#         openai_api_key=together_api_key,
#         openai_api_base="https://api.together.xyz/v1",
#         max_tokens=2048 # Increased tokens for detailed analysis
#     )
# except Exception as e:
#     st.error(f"Failed to initialize the Language Model. Please check your API key. Error: {e}")
#     llm = None # Set llm to None to prevent further errors
# Import the correct class for OpenAI-compatible APIs
from langchain_openai import ChatOpenAI

# Use a try-except block to handle potential API key errors gracefully
try:
    # Ensure you are reading the correct secret name
    together_api_key = st.secrets["TOGETHER_API_KEY"]

    # Initialize the ChatOpenAI class to connect to Together AI
    llm = ChatOpenAI(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.0,
        openai_api_key=together_api_key,
        openai_api_base="https://api.together.xyz/v1", # This is the crucial endpoint for Together AI
        max_tokens=2048
    )
except Exception as e:
    st.error(f"Failed to initialize the Language Model. Please check your API key in secrets.toml. Error: {e}")
    llm = None # Set llm to None to prevent further errors

# --- Agent Tools ---
@tool
def get_news_urls(query: str, max_results: int = 1) -> list[dict]:
    """Fetches news article URLs and titles based on a search query."""
    #st.write(f"Tool: Searching news for '{query}'...")
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, region="wt-wt", safesearch="Moderate", max_results=max_results)
            return [{'title': r['title'], 'url': r['url']} for r in results]
    except Exception as e:
        return []

@tool
def extract_article_content(url: str) -> str:
    """Extracts the main text content from a given article URL."""
    #st.write(f"Tool: Extracting content from {url}...")
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting article from {url}: {e}"

@tool
def chartink_screener(url: str, scan_clause: dict) -> pd.DataFrame:
    """Scrapes stock data from Chartink.com based on a given URL and scan clause."""
    #st.write(f"Tool: Scraping Chartink for stock data...")
    df = pd.DataFrame()
    try:
        with requests.Session() as s:
            r = s.get(url)
            soup = bs(r.text, "html.parser")
            csrf = soup.select_one("[name='csrf-token']")['content']
            s.headers['x-csrf-token'] = csrf
            s.headers['Content-Type'] = 'application/x-www-form-urlencoded'
            r = s.post('https://chartink.com/screener/process', data=scan_clause)
            data = r.json().get('data', [])
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df['per_chg'] = pd.to_numeric(df['per_chg'], errors='coerce')
            df['rsi'] = pd.to_numeric(df.get('rsi', 0), errors='coerce')
            df['close'] = pd.to_numeric(df.get('close', 0), errors='coerce')
            df = df.sort_values(by='per_chg', ascending=False)
            df['entry'] = df['close'] * 0.98
            df['target'] = df['close'] * 1.04
            df['stop_loss'] = df['close'] * 0.96
        return df
    except Exception as e:
        return pd.DataFrame()

# This function is no longer a @tool for the agent
# The @tool decorator is now removed.
def create_stock_plot(df: pd.DataFrame, strategy_name: str, is_dark_mode: bool):
    """Generates a bubble chart visualization of stock data for UI display."""
    # ... function code
    if df.empty:
        return None

    # Use the passed-in boolean to set the style
    style_to_use = 'dark_background' if is_dark_mode else 'default'
    plt.style.use(style_to_use)

    fig, ax = plt.subplots(figsize=(10, 6)) # Smaller figure for better fit

    sns.scatterplot(data=df, x='rsi', y='per_chg',
                    size='close', sizes=(50, 500),
                    alpha=0.7, ax=ax)

    title_color = 'white' if is_dark_mode else 'black'
    ax.set_title(f'Stock Performance: {strategy_name}\nBubble Size = Closing Price', color=title_color)
    ax.set_xlabel('RSI (14)', color=title_color)
    ax.set_ylabel('Percentage Change', color=title_color)
    ax.axhline(0, color='grey', linestyle='--')
    ax.axvline(50, color='grey', linestyle='--')
    ax.grid(True, alpha=0.3)

    if is_dark_mode:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    plt.tight_layout()
    return fig

def route_user_query(query: str) -> str:
    """Classifies the user's query to determine the required action."""
    print(f"Routing query: {query}")
    if not llm:
        # If the LLM isn't working, default to running the main analysis
        return "stock_analysis"

    # Prompt for the LLM to classify the user's intent
    prompt = f"""
    Analyze the user's query and classify it as either 'stock_analysis' or 'greeting'.
    Respond with only one of those two words and nothing else.

    - If the query asks to analyze stocks, find top stocks, check the market, or is a complex financial question, classify it as 'stock_analysis'.
    - If the query is a simple "hi", "hello", "thanks", or conversational filler, classify it as 'greeting'.

    User query: "{query}"
    Classification:
    """
    try:
        response = llm.invoke(prompt)
        # Clean the response to be safe
        classification = response.content.strip().lower()
        if "greeting" in classification:
            return "greeting"
        else:
            return "stock_analysis"
    except Exception as e:
        print(f"Error during routing: {e}")
        # Default to the main function if routing fails
        return "stock_analysis"

# --- Agent State ---
class GraphState(TypedDict):
    query: str
    risk_profile: str
    analysis_type: str
    news_articles: List[Dict[str, Any]]
    stock_data: Dict[str, pd.DataFrame]
    visualizations: Dict[str, Any]
    llm_recommendation: str

# --- Agent Nodes ---
def stock_screener_node(state: GraphState) -> GraphState:
    #st.write("Executing Node: StockScreener")
    strategies = {
        'Smart Money': ('https://chartink.com/screener/strategy-1-499', {'scan_clause': '( {cash} ( latest close > latest ema( latest close , 200 ) and latest close > latest ema( latest close , 50 ) and latest ema( latest close , 50 ) > latest ema( latest close , 200 ) and latest rsi( 14 ) > 50 and latest macd signal( 26 , 12 , 9 ) > 0 and latest macd line( 26 , 12 , 9 ) > 0 and latest macd line( 26 , 12 , 9 ) > latest macd signal( 26 , 12 , 9 ) ) )'}),
        'Momentum': ('https://chartink.com/screener/momentum', {'scan_clause': '( {cash} ( latest close > latest ema( latest close , 200 ) and latest rsi( 14 ) > 70 ) )'})
    }
    all_stock_data = {}
    for name, (url, clause) in strategies.items():
        df = chartink_screener.invoke({"url": url, "scan_clause": clause})
        if not df.empty:
            all_stock_data[name] = df
    return {"stock_data": all_stock_data}

def news_fetcher_node(state: GraphState) -> GraphState:
    #st.write("Executing Node: NewsFetcher")
    stock_data = state.get("stock_data", {})
    main_df = stock_data.get("Smart Money", pd.DataFrame())
    articles = []
    if not main_df.empty:
        top_stocks = main_df.head(3)['nsecode'].tolist() # Limit to 3 stocks for speed
        for symbol in top_stocks:
            news_results = get_news_urls.invoke({"query": f"{symbol} stock news", "max_results": 1})
            for article_info in news_results:
                content = extract_article_content.invoke({"url": article_info['url']})
                articles.append({"symbol": symbol, "title": article_info['title'], "content": content})
    return {"news_articles": articles}

async def analyze_news_sentiment_node(state: GraphState) -> GraphState:
    #st.write("Executing Node: NewsSentiment")
    updated_articles = []
    for article in state.get("news_articles", []):
        prompt = f"""Analyze the sentiment of this financial news article about {article['symbol']} as Positive, Neutral, or Negative. Then, recommend an investment action (Buy, Hold, Sell) with a brief reason (under 20 words). Format your response as:
Sentiment: [sentiment]
Action: [action]
Reason: [reason]

Article:
```{article['content'][:1500]}```"""
        if llm:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            sentiment_analysis = response.content
        else:
            sentiment_analysis = "Error: LLM not available."
        
        updated_articles.append({**article, "sentiment_analysis": sentiment_analysis})
    return {"news_articles": updated_articles}

async def analyze_stocks_llm_node(state: GraphState) -> GraphState:
    #st.write("Executing Node: StockAnalyzer (Final Recommendation)")
    stock_data = state.get("stock_data", {})
    main_df = stock_data.get("Smart Money", pd.DataFrame())
    news_articles = state.get("news_articles", [])
    risk_profile = state.get("risk_profile")
    analysis_type = state.get("analysis_type")

    if main_df.empty:
        return {"llm_recommendation": "No stock data found to analyze."}

    top_df = main_df.head(5)[['nsecode', 'close', 'per_chg', 'rsi']]
    stock_info = top_df.to_string(index=False)

    sentiment_summary = ""
    for article in news_articles:
        sentiment_summary += f"\n- **{article.get('symbol')}**: {article.get('sentiment_analysis', 'N/A').replace('n', ' ')}"

    prompt = f"""
You are a senior financial analyst providing a stock investment recommendation.

**Client Profile:**
- **Risk Tolerance:** {risk_profile}
- **Requested Analysis Depth:** {analysis_type}

**Data Collected:**
1.  **Top Stocks from 'Smart Money' Screener:**
    {stock_info}

2.  **News Sentiment Summary:**
    {sentiment_summary}

**Your Task:**
Based on all the data provided and keeping the client's profile in mind, provide a clear, final investment recommendation. Structure your response as follows:

1.  **Executive Summary:** A brief overview of your findings.
2.  **Top Picks (1-2 stocks):** For each stock, provide a concise justification referencing its technicals (RSI, momentum) and news sentiment.
3.  **Risk Assessment:** Briefly mention any risks or stocks to be cautious about.
4.  **Actionable Plan:** Conclude with a clear recommendation (e.g., "Recommend buying STOCK_A," "Suggest holding STOCK_B," etc.).

Your tone should be professional, data-driven, and confident.
"""
    if llm:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        recommendation = response.content
    else:
        recommendation = "Error: LLM not available for final analysis."
        
    return {"llm_recommendation": recommendation}

# --- Graph Definition ---
if llm: # Only build the graph if the LLM is available
    builder = StateGraph(GraphState)
    builder.add_node("StockScreener", stock_screener_node)
    builder.add_node("NewsFetcher", news_fetcher_node)
    builder.add_node("NewsSentiment", analyze_news_sentiment_node)
    builder.add_edge("NewsSentiment", "StockAnalyzer")
    builder.add_node("StockAnalyzer", analyze_stocks_llm_node)

    builder.set_entry_point("StockScreener")
    builder.add_edge("StockScreener", "NewsFetcher")
    builder.add_edge("NewsFetcher", "NewsSentiment") # Make sure this edge exists
    builder.add_edge("NewsSentiment", "StockAnalyzer") # The graph flows from Sentiment directly to Analyzer
    builder.add_edge("StockAnalyzer", END)
    graph = builder.compile()
else:
    graph = None

# --- Asynchronous Runner for Streamlit ---
def run_stock_analysis_agent(query: str, risk_profile: str, analysis_type: str):
    """Wraps the async graph execution for the sync Streamlit environment."""
    if not graph:
        return {"llm_recommendation": "Analysis agent could not be initialized. Please check API key configuration."}

    async def _run_graph():
        initial_state = {
            "query": query,
            "risk_profile": risk_profile,
            "analysis_type": analysis_type,
        }
        return await graph.ainvoke(initial_state)

    # Run the async function from a sync context
    return asyncio.run(_run_graph())


# ==============================================================================
# --- STREAMLIT FRONTEND ---
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Management ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# The rest of your CSS and theme management code remains unchanged...
# [I've omitted the large CSS block for brevity, but you should paste your original `apply_theme_css` function and related code here]
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            "primary_bg": "#1e1e1e", "secondary_bg": "#2d2d2d", "card_bg": "#3a3a3a",
            "text_primary": "#ffffff", "text_secondary": "#b0b0b0", "accent_color": "#667eea",
            "accent_secondary": "#764ba2", "border_color": "#4a4a4a", "chat_user_bg": "#2c3e50",
            "chat_assistant_bg": "#34495e", "button_gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
            "header_gradient": "linear-gradient(90deg, #2c3e50 0%, #3498db 100%)", "shadow": "0 2px 10px rgba(0,0,0,0.3)"
        }
    else:
        return {
            "primary_bg": "#f8fafc", "secondary_bg": "#ffffff", "card_bg": "#ffffff",
            "text_primary": "#1a202c", "text_secondary": "#4a5568", "accent_color": "#667eea",
            "accent_secondary": "#764ba2", "border_color": "#e2e8f0", "chat_user_bg": "#e6f3ff",
            "chat_assistant_bg": "#f3e8ff", "button_gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
            "header_gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)", "shadow": "0 2px 8px rgba(0,0,0,0.06)"
        }

def apply_theme_css():
    colors = get_theme_colors()
    st.markdown(f"""
    <style>
        /* Global Styles */
        .stApp {{ background-color: {colors['primary_bg']}; color: {colors['text_primary']}; }}
        /* Main Header */
        .main-header {{ background: {colors['header_gradient']}; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center; box-shadow: {colors['shadow']}; }}
        .main-header h1, .main-header p {{ color: white !important; }}
        /* Sidebar */
        section[data-testid="stSidebar"] {{ background-color: {colors['secondary_bg']} !important; border-right: 2px solid {colors['border_color']}; }}
        /* Chat Messages */
        .stChatMessage[data-testid="chat-message-user"] {{ background-color: {colors['chat_user_bg']}; border-radius: 12px; border: 1px solid {colors['border_color']}; }}
        .stChatMessage[data-testid="chat-message-assistant"] {{ background-color: {colors['chat_assistant_bg']}; border-radius: 12px; border: 1px solid {colors['border_color']}; }}
        /* Other component styles - you can add more from your original code as needed */
        .stButton > button {{ background: {colors['button_gradient']}; color: white; border: none; border-radius: 25px; padding: 0.5rem 2rem; font-weight: bold; }}
        .stMetric {{ background-color: {colors['card_bg']}; padding: 1rem; border-radius: 10px; box-shadow: {colors['shadow']}; border: 1px solid {colors['border_color']}; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme_css()


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered Stock Advisor Agent</h1>
    <p>Get intelligent stock analysis powered by a multi-agent system</p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<h3>🎛️ Agent Configuration</h3>", unsafe_allow_html=True)
    
    analysis_depth = st.selectbox(
        "Analysis Depth",
        ["Quick", "Standard", "Deep"], index=1
    )
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"], value="Moderate"
    )
    
    st.markdown("---")
    st.markdown("<h3>📊 Session Metrics</h3>", unsafe_allow_html=True)
    queries_processed = len([msg for msg in st.session_state.get('messages', []) if msg['role'] == 'user'])
    st.metric("Queries Processed", queries_processed)


# --- Main Content Area ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI Stock Advisor. Ask me to analyze the market (e.g., 'Find top momentum stocks') to begin."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # The content of assistant messages can be complex (strings, dataframes, plots)
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        elif isinstance(message["content"], dict): # Handle the complex agent response
             # Render the dictionary content in a structured way
            if "recommendation" in message["content"]:
                st.markdown(message["content"]["recommendation"])
            if message["content"].get("sentiments"):
                st.markdown("--- \n### 📰 News Sentiment Analysis")
                for item in message["content"]["sentiments"]:
                    st.info(f"**{item.get('symbol', 'N/A')}**: {item.get('title', 'N/A')}")
                    st.text(item.get('sentiment_analysis', 'N/A'))
            if message["content"].get("strategies"):
                st.markdown("--- \n### 📈 Strategy Screener Results")
                for name, df in message["content"]["strategies"].items():
                    st.subheader(f"Strategy: {name}")
                    st.dataframe(df.head()) # Show top 5 results
            if message["content"].get("visualizations"):
                st.markdown("--- \n### 📊 Visualizations")
                for name, fig in message["content"]["visualizations"].items():
                    st.pyplot(fig)


# Accept user input
# Accept user input
if prompt := st.chat_input("e.g., 'Analyze the market for top stocks'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # First, determine the user's intent
        intent = route_user_query(prompt)

        # --- Conditional Logic Based on Intent ---
        if intent == "stock_analysis":
            with st.spinner("🤖 The AI agent is analyzing the market... This may take a moment."):
                agent_results = run_stock_analysis_agent(
                    query=prompt,
                    risk_profile=risk_tolerance,
                    analysis_type=analysis_depth
                )

            # Prepare a dictionary to hold all parts of the response for history
            response_content_for_history = {
                "recommendation": agent_results.get("llm_recommendation", "No recommendation available."),
                "sentiments": agent_results.get("news_articles", []),
                "strategies": agent_results.get("stock_data", {})
            }

            # --- ADDED: Display Common Stocks Across Strategies ---
            strategies_data = response_content_for_history.get("strategies", {})
            if len(strategies_data) > 1:
                try:
                    stock_sets = [set(df['nsecode']) for df in strategies_data.values()]
                    common_stocks = set.intersection(*stock_sets)
                    if common_stocks:
                        st.success(f"🔥 **Cross-Strategy Confirmation:** The following stocks appeared in ALL strategies: **{', '.join(common_stocks)}**")
                except Exception as e:
                    print(f"Could not calculate common stocks: {e}")


            # Display the main recommendation
            if response_content_for_history["recommendation"]:
                st.markdown(response_content_for_history["recommendation"])

            # --- UPGRADED: Display Sentiments with Clickable Links ---
            if response_content_for_history["sentiments"]:
                st.markdown("--- \n### 📰 News Sentiment Analysis")
                for item in response_content_for_history["sentiments"]:
                    title = item.get('title', 'N/A')
                    url = item.get('url', '#')
                    st.info(f"**[{item.get('symbol', 'N/A')}]**: [{title}]({url})") # Title is now a link
                    st.text(item.get('sentiment_analysis', 'N/A'))

            # --- UPGRADED: Display Strategies with Risk/Reward Ratio ---
            if response_content_for_history["strategies"]:
                st.markdown("--- \n### 📈 Strategy Screener Results & Plots")
                for name, df in response_content_for_history["strategies"].items():
                    st.subheader(f"Strategy: {name}")

                    # Calculate RR Ratio, handling potential division by zero
                    if all(k in df for k in ['entry', 'stop_loss', 'target']):
                        entry_diff = df['entry'] - df['stop_loss']
                        target_diff = df['target'] - df['entry']
                        df['RR Ratio'] = target_diff.div(entry_diff.where(entry_diff != 0, float('nan'))).round(2)
                        st.dataframe(df[['nsecode', 'close', 'entry', 'target', 'stop_loss', 'RR Ratio']].head())
                    else:
                        st.dataframe(df.head()) # Fallback if columns are missing

                    # Create and display the plot
                    st.markdown("#### Performance Visualization")
                    fig = create_stock_plot(df, name, st.session_state.dark_mode)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.write("Not enough data to create a plot.")

            # Add the complete assistant response dictionary to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})

        else: # This handles the 'greeting' intent
            response_text = "Hello! I am an AI stock analysis agent. Please ask me to analyze the market or find top stocks."
            st.markdown(response_text)
            # Add the simple greeting to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚠️ <strong>Disclaimer</strong>: This is an AI-driven analysis. All data and recommendations are for informational purposes only.</p>
    <p>🔒 Always conduct your own research before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)