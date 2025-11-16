# Quick Start Guide

## 1. Environment Setup

You can do this via our setup script
```bash
bash setup.sh
```

Or, you can set up your environment manually.
```bash
# Navigate to project directory
cd agentics-finance

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure API Key

Create a `.env` file in the project root:

```bash
# Copy the example
cp env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

## 3. Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 4. Try Example Questions

Once the app is running, try asking:

### Basic Queries
- "List all available indicators"
- "Show me data for S&P 500 from 2020-01-01 to 2020-12-31"
- "What are the statistics for the VIX index?"

### Volatility Analysis
- "What was the volatility of the S&P 500 during March 2020?"
- "Find the periods of highest volatility for Bitcoin"
- "Show me a volatility plot for crude oil prices in 2022"

### Correlation Analysis
- "What's the correlation between S&P 500 and VIX?"
- "Show me a correlation heatmap for ^GSPC, ^VIX, GLD, and BTC-USD"
- "Analyze the relationship between unemployment and the stock market"

### Visualizations
- "Plot the Federal Funds Rate from 2008 to present"
- "Create a time series plot comparing S&P 500 and gold prices"
- "Show me the distribution of unemployment rates since 2008"

### Complex Analysis
- "What happened to the stock market during the 2008 financial crisis?"
- "Compare inflation, unemployment, and interest rates during COVID-19"
- "Find the dates when oil prices reached their peak values"

## 5. Understanding the Output

The agent will:
1. **Analyze your question** to determine what data and tools are needed
2. **Use appropriate tools** to query and analyze the data
3. **Generate visualizations** if helpful for your question
4. **Provide a comprehensive answer** with context and insights

Visualizations are displayed directly in the chat interface and are automatically cleaned up between conversation turns.

## 6. Troubleshooting

### "No module named 'crewai'"
Make sure you've activated the virtual environment and installed requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "OpenAI API key not found"
Check that your `.env` file exists and contains a valid API key:
```bash
cat .env
```

### "Data files not found"
Ensure the CSV files are in the correct location:
```bash
ls data/
# Should show: macro_factors_new.csv  market_factors_new.csv
```

### Agent is not responding or taking too long
- Check your internet connection (required for OpenAI API)
- Verify your OpenAI API key has sufficient credits
- Try asking a simpler question first
- Check the terminal for any error messages

## 7. Tips for Best Results

1. **Be specific**: Mention exact indicator names and date ranges when possible
2. **Use proper date format**: YYYY-MM-DD (e.g., 2020-01-01)
3. **Ask one thing at a time**: Complex questions can be broken down into steps
4. **Request visualizations**: The agent creates better visuals when you explicitly ask
5. **Provide context**: Mention if you want analysis for a specific event (e.g., "during the 2008 crisis")

## 8. Architecture Overview

```
User Question (Streamlit UI)
    ↓
CrewAI Agent (GPT-4)
    ↓
Uses Tools:
├── Data Query Tools (fetch data)
├── Analysis Tools (calculate statistics, volatility, correlation)
└── Visualization Tools (create charts)
    ↓
Agent Response + Visualizations
    ↓
Displayed in Streamlit UI
```

## 9. Next Steps

- Try different types of questions to explore the data
- Experiment with different date ranges and indicators
- Combine multiple indicators in correlations and comparisons
- Ask follow-up questions in the same conversation

Enjoy exploring the financial data! 📊

