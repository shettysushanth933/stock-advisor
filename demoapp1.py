import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

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

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            "primary_bg": "#1e1e1e",
            "secondary_bg": "#2d2d2d",
            "card_bg": "#3a3a3a",
            "text_primary": "#ffffff",
            "text_secondary": "#b0b0b0",
            "accent_color": "#667eea",
            "accent_secondary": "#764ba2",
            "border_color": "#4a4a4a",
            "chat_user_bg": "#2c3e50",
            "chat_assistant_bg": "#34495e",
            "button_gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
            "header_gradient": "linear-gradient(90deg, #2c3e50 0%, #3498db 100%)",
            "shadow": "0 2px 10px rgba(0,0,0,0.3)"
        }
    else:
        return {
            "primary_bg": "#f8fafc",  # slightly cooler background
            "secondary_bg": "#ffffff",  # pure white for cards/sidebars
            "card_bg": "#ffffff",
            "text_primary": "#1a202c",  # strong dark text for better readability
            "text_secondary": "#4a5568",  # medium gray for secondary text
            "accent_color": "#667eea",
            "accent_secondary": "#764ba2",
            "border_color": "#e2e8f0",  # softer border
            "chat_user_bg": "#e6f3ff",  # lighter blue for user
            "chat_assistant_bg": "#f3e8ff",  # lighter purple for assistant
            "button_gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
            "header_gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
            "shadow": "0 2px 8px rgba(0,0,0,0.06)"  # even softer shadow
        }

# --- Custom CSS Styling with Theme Support ---
def apply_theme_css():
    colors = get_theme_colors()
    
    # Compute theme-dependent colors for chat and input
    user_msg_bg = colors['chat_user_bg']
    assistant_msg_bg = colors['chat_assistant_bg']
    # Strong contrast text colors for both themes
    chat_text_color = colors['text_primary']
    chat_input_bg = colors['card_bg']
    chat_input_text = colors['text_primary']
    toggle_color = '#ffffff' if st.session_state.dark_mode else '#1a202c'

    st.markdown(f"""
    <style>
        /* Global Theme Variables */
        .stApp {{
            background-color: {colors['primary_bg']};
            color: {colors['text_primary']};
        }}
        
        /* Force text color override for all elements */
        .stApp *, .stApp *::before, .stApp *::after {{
            color: {colors['text_primary']} !important;
        }}
        
        /* Main Header */
        .main-header {{
            background: {colors['header_gradient']};
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            color: white !important;
            box-shadow: {colors['shadow']};
        }}
        
        .main-header h1, .main-header p {{
            color: white !important;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: {colors['card_bg']};
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: {colors['shadow']};
            border-left: 5px solid {colors['accent_color']};
            margin-bottom: 1rem;
            color: {colors['text_primary']};
            border: 1px solid {colors['border_color']};
        }}
        
        /* Agent Status */
        .agent-status {{
            background: {colors['button_gradient']};
            color: white !important;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: {colors['shadow']};
        }}
        
        /* Chat Container */
        .chat-container {{
            background: {colors['secondary_bg']};
            border-radius: 12px;
            padding: 1.5rem;
            min-height: 400px;
            margin-bottom: 1rem;
            box-shadow: {colors['shadow']};
            border: 1px solid {colors['border_color']};
        }}
        
        /* Message Styling with Improved Contrast */
        .user-message {{
            background: {user_msg_bg} !important;
            padding: 1rem 1.25rem;
            border-radius: 16px;
            margin: 0.75rem 0;
            border-left: 4px solid {colors['accent_color']};
            color: {chat_text_color} !important;
            box-shadow: {colors['shadow']};
            border: 1px solid {colors['border_color']};
        }}
        
        .user-message * {{
            color: {chat_text_color} !important;
        }}
        
        .assistant-message {{
            background: {assistant_msg_bg} !important;
            padding: 1rem 1.25rem;
            border-radius: 16px;
            margin: 0.75rem 0;
            border-left: 4px solid {colors['accent_secondary']};
            color: {chat_text_color} !important;
            box-shadow: {colors['shadow']};
            border: 1px solid {colors['border_color']};
        }}
        
        .assistant-message * {{
            color: {chat_text_color} !important;
        }}
        
        /* Streamlit Chat Message Override */
        .stChatMessage {{
            color: {chat_text_color} !important;
        }}
        
        .stChatMessage * {{
            color: {chat_text_color} !important;
        }}
        
        .stChatMessage[data-testid="chat-message-user"] {{
            background-color: {user_msg_bg} !important;
            border: 1px solid {colors['border_color']};
        }}
        
        .stChatMessage[data-testid="chat-message-assistant"] {{
            background-color: {assistant_msg_bg} !important;
            border: 1px solid {colors['border_color']};
        }}
        
        /* Button Styling */
        .stButton > button {{
            background: {colors['button_gradient']} !important;
            color: white !important;
            border: none !important;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: {colors['shadow']};
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        }}
        
        /* Sidebar Styling */
        .sidebar-content {{
            background: {colors['card_bg']};
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: {colors['shadow']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_color']};
        }}
        
        .sidebar-content h3 {{
            color: {colors['text_primary']} !important;
            margin-bottom: 1rem;
        }}
        
        /* Streamlit Components Theme Override */
        .stSelectbox > div > div {{
            background-color: {colors['card_bg']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border_color']};
        }}
        
        .stSelectbox label {{
            color: {colors['text_primary']} !important;
        }}
        
        .stCheckbox > label {{
            color: {colors['text_primary']} !important;
        }}
        
        .stCheckbox > label > span {{
            color: {colors['text_primary']} !important;
        }}
        
        .stSlider > label {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMetric {{
            background-color: {colors['card_bg']} !important;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: {colors['shadow']};
            border: 1px solid {colors['border_color']};
        }}
        
        .stMetric > div {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMetric label {{
            color: {colors['text_secondary']} !important;
        }}
        
        .stExpander {{
            background-color: {colors['card_bg']} !important;
            border: 1px solid {colors['border_color']};
            border-radius: 8px;
        }}
        
        .stExpander > div > div {{
            color: {colors['text_primary']} !important;
        }}
        
        .stExpander summary {{
            color: {colors['text_primary']} !important;
            background-color: {colors['card_bg']} !important;
        }}
        
        /* Chat Input Styling */
        .stChatInput > div {{
            background-color: {chat_input_bg} !important;
            border: 1px solid {colors['border_color']};
            border-radius: 25px;
        }}
        
        .stChatInput input {{
            color: {chat_input_text} !important;
            background-color: {chat_input_bg} !important;
        }}
        
        .stChatInput input::placeholder {{
            color: {colors['text_secondary']} !important;
        }}
        
        /* Theme Toggle Button */
        .theme-toggle {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 999;
            background: {colors['button_gradient']};
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            color: {toggle_color} !important;
            cursor: pointer;
            box-shadow: {colors['shadow']};
            transition: all 0.3s ease;
            font-size: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .theme-toggle:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        }}
        
        /* Sidebar background override */
        section[data-testid="stSidebar"] {{
            background-color: {colors['secondary_bg']} !important;
            border-right: 2px solid {colors['border_color']};
        }}
        
        section[data-testid="stSidebar"] * {{
            color: {colors['text_primary']} !important;
        }}
        
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {colors['text_primary']} !important;
        }}

        /* Main content area background */
        div[data-testid="stAppViewContainer"] {{
            background-color: {colors['primary_bg']} !important;
        }}
        
        /* Info/Warning Box Styling */
        .stInfo, .stSuccess, .stWarning, .stError {{
            border-radius: 8px;
            border: 1px solid {colors['border_color']};
        }}
        
        .stInfo > div {{
            color: {colors['text_primary']} !important;
            background-color: {colors['card_bg']} !important;
        }}
        
        /* JSON Display Styling */
        .stJson {{
            background-color: {colors['card_bg']} !important;
            border: 1px solid {colors['border_color']};
            border-radius: 8px;
        }}
        
        /* Markdown Content */
        .stMarkdown {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMarkdown p {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMarkdown ul, .stMarkdown ol {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMarkdown li {{
            color: {colors['text_primary']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Apply theme
apply_theme_css()

# --- Header with Theme Toggle ---
col_header, col_toggle = st.columns([10, 1])

with col_header:
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI-Powered Stock Advisor Agent</h1>
        <p>Get intelligent stock analysis powered by advanced data-aware agents</p>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    # Theme toggle button (icon only)
    theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
    if st.button(theme_icon, key="theme_toggle", help="Toggle Dark/Light Mode"):
        toggle_theme()
        st.rerun()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-content">
        <h3>🎛️ Agent Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme status indicator
    theme_status = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
    st.info(f"Current Theme: {theme_status}")
    
    # Agent Settings
    with st.expander("🔧 Agent Settings", expanded=True):
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Analysis", "Standard Analysis", "Deep Analysis"],
            index=1
        )
        
        include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
        include_technical = st.checkbox("Include Technical Indicators", value=True)
        include_news = st.checkbox("Include Recent News", value=True)
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
    
    # Agent State Display
    st.markdown(f"""
    <div class="sidebar-content">
        <h3>🧠 Agent Internal State</h3>
    </div>
    """, unsafe_allow_html=True)
    
    agent_state = {
        "status": "Ready",
        "theme": "Dark Mode" if st.session_state.dark_mode else "Light Mode",
        "last_analysis": "None",
        "data_sources": ["Yahoo Finance", "News API", "Technical Analysis"],
        "memory_entries": 0,
        "active_tools": ["stock_data", "sentiment_analyzer", "technical_analyzer"]
    }
    
    with st.expander("View Agent State", expanded=False):
        st.json(agent_state)
    
    # Performance Metrics
    st.markdown(f"""
    <div class="sidebar-content">
        <h3>📊 Session Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries Processed", len(st.session_state.get('messages', [])) // 2)
    with col2:
        st.metric("Success Rate", "100%")

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat with Your Stock Advisor")
    
    # Initialize chat history (similar to original)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"type": "assistant", "content": "Hello! I'm your advanced Stock Advisor (frontend demo). What stock are you interested in?"}
        ]
    
    # Display chat messages (using streamlit's built-in chat UI)
    for message in st.session_state.messages:
        with st.chat_message(message["type"]):
            st.write(message["content"])
    
    # Chat input (similar to original logic)
    if prompt := st.chat_input("Ask me about a stock..."):
        # Add user message
        user_message = {"type": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        with st.chat_message(user_message["type"]):
            st.write(user_message["content"])
        
        # Generate placeholder assistant response (similar to original)
        assistant_message = {"type": "assistant", "content": f"(No backend: This is a placeholder response for '{prompt}'. In a real implementation, the agent would analyze the stock data and provide insights.)"}
        st.session_state.messages.append(assistant_message)
        
        with st.chat_message(assistant_message["type"]):
            st.write(assistant_message["content"])

with col2:
    st.markdown("### 📈 Quick Market Overview")
    
    # Mock market data with theme-aware styling
    market_data = {
        "S&P 500": {"price": 4567.89, "change": "+1.23%", "color": "green"},
        "NASDAQ": {"price": 14234.56, "change": "+0.87%", "color": "green"},
        "DOW": {"price": 34567.12, "change": "-0.45%", "color": "red"}
    }
    
    for index, data in market_data.items():
        delta_color = "normal" if data["color"] == "green" else "inverse"
        st.metric(
            index, 
            f"{data['price']:,.2f}", 
            data['change'],
            delta_color=delta_color
        )
    
    st.markdown("### 🔥 Trending Stocks")
    trending_stocks = [
        ("AAPL", 2.15),
        ("TSLA", -1.32),
        ("GOOGL", 0.87),
        ("MSFT", 1.05),
        ("AMZN", -0.44)
    ]
    for stock, change in trending_stocks:
        color = "🟢" if change > 0 else "🔴"
        st.write(f"{color} **{stock}** {change:+.2f}%")
    
    st.markdown("### 🎯 Analysis Features")
    features = [
        "📊 Real-time data analysis",
        "📰 News sentiment tracking", 
        "📈 Technical indicators",
        "⚠️ Risk assessment",
        "🎯 Price predictions",
        "📱 Mobile-friendly interface",
        f"🎨 {'Dark' if st.session_state.dark_mode else 'Light'} mode support"
    ]
    
    for feature in features:
        st.write(feature)

# --- Enhanced Quick Action Buttons (Commented Out) ---
# Uncomment the section below to enable detailed responses for quick actions

# with col1:
#     if st.button("📈 Market Summary"):
#         st.session_state.messages.append({
#             "type": "user",
#             "content": "Give me a market summary",
#             "timestamp": datetime.now()
#         })
#         st.session_state.messages.append({
#             "type": "assistant",
#             "content": "📊 **Market Summary**: Today's market showed mixed signals with tech stocks leading gains while energy sectors declined. Key highlights include strong earnings from major tech companies and concerns about inflation data.",
#             "timestamp": datetime.now()
#         })
#         st.rerun()

# with col2:
#     if st.button("🔍 Popular Stocks"):
#         popular_analysis = """
#         📈 **Popular Stocks Analysis**:
#         
#         • **AAPL**: Strong momentum, trading above key support
#         • **TSLA**: Volatile but showing bullish patterns
#         • **GOOGL**: Consolidating near resistance levels
#         • **MSFT**: Steady uptrend with strong fundamentals
#         """
#         st.session_state.messages.append({
#             "type": "user",
#             "content": "Show me popular stocks",
#             "timestamp": datetime.now()
#         })
#         st.session_state.messages.append({
#             "type": "assistant",
#             "content": popular_analysis,
#             "timestamp": datetime.now()
#         })
#         st.rerun()

# with col3:
#     if st.button("⚡ Quick Tips"):
#         tips = """
#         💡 **Trading Tips**:
#         
#         • Always diversify your portfolio
#         • Set stop-loss orders to manage risk
#         • Keep up with earnings calendars
#         • Consider market sentiment in decisions
#         • Never invest more than you can afford to lose
#         """
#         st.session_state.messages.append({
#             "type": "user",
#             "content": "Give me some trading tips",
#             "timestamp": datetime.now()
#         })
#         st.session_state.messages.append({
#             "type": "assistant",
#             "content": tips,
#             "timestamp": datetime.now()
#         })
#         st.rerun()

# with col4:
#     if st.button("🔄 Clear Chat"):
#         st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
#         st.rerun()

# --- Quick Action Buttons ---
st.markdown("### 🚀 Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📈 Market Summary"):
        st.session_state.messages.append({"type": "user", "content": "Give me a market summary"})
        st.session_state.messages.append({"type": "assistant", "content": "(No backend: This would show a comprehensive market summary with key indices and trends.)"})
        st.rerun()

with col2:
    if st.button("🔍 Popular Stocks"):
        st.session_state.messages.append({"type": "user", "content": "Show me popular stocks"})
        st.session_state.messages.append({"type": "assistant", "content": "(No backend: This would display analysis of trending stocks like AAPL, TSLA, GOOGL, etc.)"})
        st.rerun()

with col3:
    if st.button("⚡ Quick Tips"):
        st.session_state.messages.append({"type": "user", "content": "Give me some trading tips"})
        st.session_state.messages.append({"type": "assistant", "content": "(No backend: This would provide personalized trading tips and risk management advice.)"})
        st.rerun()

with col4:
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = [{"type": "assistant", "content": "Hello! I'm your advanced Stock Advisor (frontend demo). What stock are you interested in?"}]
        st.rerun()

# --- Advanced Response Generation (Commented Out for Demo) ---
# Uncomment the function below to enable dynamic responses based on user settings

# def generate_mock_response(prompt, depth, sentiment, technical):
#     """Generate a mock response based on user input and settings"""
#     
#     # Extract stock symbol if mentioned
#     import re
#     stock_match = re.search(r'\b[A-Z]{2,5}\b', prompt.upper())
#     stock_symbol = stock_match.group() if stock_match else "STOCK"
#     
#     base_response = f"🎯 **Analysis for {stock_symbol}**:\n\n"
#     
#     if depth == "Quick Analysis":
#         response = base_response + f"""
#         📊 **Current Status**: {stock_symbol} is trading within normal ranges with moderate volume.
#         
#         💡 **Quick Insight**: Based on recent data, the stock shows {np.random.choice(['bullish', 'neutral', 'cautious'])} signals.
#         """
#     
#     elif depth == "Standard Analysis":
#         response = base_response + f"""
#         📊 **Price Action**: {stock_symbol} is currently at ${np.random.uniform(50, 300):.2f} with {np.random.choice(['strong', 'moderate', 'weak'])} momentum.
#         
#         📈 **Technical Signals**: 
#         • RSI: {np.random.randint(30, 70)}
#         • Moving Average: {np.random.choice(['Above', 'Below'])} 50-day MA
#         
#         📰 **Sentiment**: {np.random.choice(['Positive', 'Neutral', 'Mixed'])} based on recent news.
#         
#         💡 **Recommendation**: {np.random.choice(['BUY', 'HOLD', 'WATCH'])} with {risk_tolerance.lower()} risk approach.
#         """
#     
#     else:  # Deep Analysis
#         response = base_response + f"""
#         📊 **Comprehensive Analysis**:
#         
#         **Price Metrics**:
#         • Current: ${np.random.uniform(50, 300):.2f}
#         • 52W High: ${np.random.uniform(300, 400):.2f}
#         • 52W Low: ${np.random.uniform(30, 80):.2f}
#         • Volume: {np.random.uniform(1, 50):.1f}M shares
#         
#         **Technical Indicators**:
#         • RSI(14): {np.random.randint(30, 70)}
#         • MACD: {np.random.choice(['Bullish', 'Bearish'])} crossover
#         • Bollinger Bands: {np.random.choice(['Expanding', 'Contracting'])}
#         
#         **Fundamental Data**:
#         • P/E Ratio: {np.random.uniform(15, 35):.1f}
#         • Market Cap: ${np.random.uniform(10, 500):.1f}B
#         
#         **Risk Assessment**: {risk_tolerance} risk profile suggests {np.random.choice(['suitable', 'proceed with caution', 'high potential'])}.
#         
#         💡 **Strategic Recommendation**: Detailed analysis suggests {np.random.choice(['accumulate on dips', 'hold current position', 'consider profit taking'])}.
#         """
#     
#     return response

# --- Alternative Chat Implementation (Commented Out) ---
# Uncomment the section below to enable advanced chat with timestamps and custom styling

# with col1:
#     st.markdown("### 💬 Chat with Your Stock Advisor")
#     
#     # Initialize chat history with advanced welcome message
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {
#                 "type": "assistant", 
#                 "content": "👋 Hello! I'm your advanced AI Stock Advisor. I can provide comprehensive stock analysis including:\n\n• **Real-time stock data** and price movements\n• **Technical analysis** with key indicators\n• **Sentiment analysis** from news and social media\n• **Risk assessment** and investment recommendations\n\nWhich stock would you like me to analyze today?",
#                 "timestamp": datetime.now()
#             }
#         ]
#     
#     # Chat container with custom styling
#     chat_container = st.container()
#     
#     with chat_container:
#         for i, message in enumerate(st.session_state.messages):
#             if message["type"] == "user":
#                 st.markdown(f"""
#                 <div class="user-message">
#                     <strong>You:</strong> {message['content']}
#                     <small style="float: right; color: #666;">
#                         {message.get('timestamp', datetime.now()).strftime('%H:%M')}
#                     </small>
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                 <div class="assistant-message">
#                     <strong>AI Advisor:</strong> {message['content']}
#                     <small style="float: right; color: #666;">
#                         {message.get('timestamp', datetime.now()).strftime('%H:%M')}
#                     </small>
#                 </div>
#                 """, unsafe_allow_html=True)
#     
#     # Chat input with advanced response generation
#     if prompt := st.chat_input("Ask me about any stock (e.g., 'Analyze AAPL' or 'What's the outlook for Tesla?')"):
#         # Add user message with timestamp
#         user_message = {
#             "type": "user", 
#             "content": prompt,
#             "timestamp": datetime.now()
#         }
#         st.session_state.messages.append(user_message)
#         
#         # Generate intelligent mock response based on settings
#         mock_response = generate_mock_response(prompt, analysis_depth, include_sentiment, include_technical)
#         
#         assistant_message = {
#             "type": "assistant", 
#             "content": mock_response,
#             "timestamp": datetime.now()
#         }
#         st.session_state.messages.append(assistant_message)
#         
#         st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚠️ <strong>Disclaimer</strong>: This is a frontend demonstration. All data and recommendations are simulated for demo purposes only.</p>
    <p>🔒 Always conduct your own research before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)