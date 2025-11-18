"""
Deep Research Analyst Agent with Code Execution Capabilities.

This agent can:
- Decompose complex problems into manageable steps
- Write and execute Python code dynamically
- Access and analyze financial datasets
- Generate visualizations
- Provide comprehensive, multi-step analysis
"""

from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv
from tools.code_execution_tool import execute_python_code, save_plotly_figure

load_dotenv()


def get_dataset_context():
    """
    Provide comprehensive context about available datasets for Deep Research.
    Dynamically reads actual column names to ensure accuracy.
    """
    import pandas as pd

    context_parts = []
    context_parts.append("AVAILABLE DATASETS:")
    context_parts.append("=" * 60)
    context_parts.append("")

    # 1. Macroeconomic Indicators
    try:
        macro_df = pd.read_csv('./data/macro_factors_new.csv', nrows=1)
        macro_cols = ', '.join(macro_df.columns.tolist())
        context_parts.append("1. MACROECONOMIC INDICATORS (./data/macro_factors_new.csv)")
        context_parts.append(f"   Columns: {macro_cols}")
        context_parts.append("   Date Range: 2008-2025")
        context_parts.append("   Key Metrics: Fed Funds Rate, CPI, Unemployment, Industrial Production")
        context_parts.append("")
    except Exception as e:
        context_parts.append(f"1. MACROECONOMIC INDICATORS - Error: {e}")
        context_parts.append("")

    # 2. Market Factors
    try:
        market_df = pd.read_csv('./data/market_factors_new.csv', nrows=1)
        market_cols = ', '.join(market_df.columns.tolist())
        context_parts.append("2. MARKET FACTORS (./data/market_factors_new.csv)")
        context_parts.append(f"   Columns: {market_cols}")
        context_parts.append("   Date Range: 2008-2025")
        context_parts.append("   Key Metrics: S&P 500, VIX, Bitcoin, Gold, Treasury Yields")
        context_parts.append("")
    except Exception as e:
        context_parts.append(f"2. MARKET FACTORS - Error: {e}")
        context_parts.append("")

    # 3. Company Fundamentals
    try:
        firm_df = pd.read_csv('./data/firm_summary.csv', nrows=1)
        firm_cols = ', '.join(firm_df.columns.tolist())
        context_parts.append("3. COMPANY FUNDAMENTALS (./data/firm_summary.csv)")
        context_parts.append(f"   Columns: {firm_cols}")
        context_parts.append("   Date Range: Quarterly data 2008-2023")
        context_parts.append("   Key Metrics: EPS, ROE, ROA, Gross Margin, Forward Growth Estimates")
        context_parts.append("")
    except Exception as e:
        context_parts.append(f"3. COMPANY FUNDAMENTALS - Error: {e}")
        context_parts.append("")

    # 4. DJ30 Stock Prices
    try:
        dj30_df = pd.read_csv('./data/dj30_data_full.csv', nrows=1)
        dj30_cols = ', '.join(dj30_df.columns.tolist())
        context_parts.append("4. DJ30 STOCK PRICES (./data/dj30_data_full.csv)")
        context_parts.append(f"   Columns: {dj30_cols}")
        context_parts.append("   Date Range: Daily data 2008-2025")
        context_parts.append("   Tickers: AAPL, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, DOW, GS,")
        context_parts.append("            HD, HON, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT,")
        context_parts.append("            NKE, PG, TRV, UNH, V, VZ, WBA, WMT")
        context_parts.append("   Key Metrics: OHLCV prices, fundamentals (PE, ROE, dividends)")
        context_parts.append("")
    except Exception as e:
        context_parts.append(f"4. DJ30 STOCK PRICES - Error: {e}")
        context_parts.append("")

    context = "\n".join(context_parts) + """
CODE EXECUTION TOOLS:
=====================
1. execute_python_code(code: str)
   - Execute Python code and return output
   - Pre-imported: pandas as pd, numpy as np, matplotlib.pyplot as plt, plotly.graph_objects as go
   - Pre-imported: save_figure(fig, filename) - saves Plotly figures directly!
   - Datasets in: './data/' directory
   - Output limit: 1800 characters (print summaries, not full dataframes)

2. save_plotly_figure(figure_json: str, filename: str = None)
   - Alternative tool for saving visualizations (if needed)
   - Takes JSON string as input

CREATING VISUALIZATIONS (EASY WAY):
====================================
Inside execute_python_code(), you can use save_figure() directly:

```python
# Create your Plotly figure
fig = go.Figure(...)
fig.update_layout(title="My Analysis")

# Save it directly (returns Visualization ID)
viz_id = save_figure(fig, "my_analysis_chart")
print(f"Saved: {viz_id}")  # Will print something like "viz_20251118_123456_abc123"
```

The save_figure() function is available inside execute_python_code() - just call it!

BEST PRACTICES:
===============
✓ Always convert dates: pd.to_datetime(df['Date'])
✓ Handle missing data: df.dropna() or df.fillna()
✓ Check data types before operations: df.dtypes
✓ Print summaries, not raw data: df.describe(), df.head(), aggregations
✓ Use try-except for robust code
✓ For visualizations, reference the Visualization ID in your final response
✗ DO NOT save pickle files (.pkl) - use only CSV or in-memory processing
✗ DO NOT use df.to_pickle() or pickle.dump()
"""

    return context


def create_deep_research_agent():
    """
    Create a Deep Research agent with code execution capabilities.

    This agent can write and execute Python code to perform complex analysis,
    decompose problems into steps, and generate comprehensive reports.
    """
    # Initialize LLM
    selected_llm = os.getenv("SELECTED_LLM", "gemini")

    if selected_llm == "gemini":
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash-exp")
    elif selected_llm == "openai":
        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4")
    else:
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash-exp")

    llm = LLM(model=model_id)

    # Use custom code execution tools (more reliable than CodeInterpreterTool)
    tools = [execute_python_code, save_plotly_figure]
    print("✅ Custom code execution tools initialized")

    agent = Agent(
        role="Senior Deep Research Financial Analyst",
        goal=(
            "Perform comprehensive financial analysis by executing Python code to calculate metrics, "
            "generate visualizations, and provide detailed, data-driven insights."
        ),
        backstory=(
            "You are an expert quantitative financial analyst with deep expertise in Python, data science, "
            "and financial modeling. You conduct rigorous analysis by executing code to calculate fundamentals, "
            "momentum indicators, risk-adjusted returns, and other key metrics. You create insightful visualizations "
            "and provide detailed explanations backed by specific numbers and statistics."
            "\n\n🚨 CRITICAL BEHAVIORAL RULES:\n"
            "1. You ALWAYS use execute_python_code() tool to perform analysis - NEVER just write code without executing\n"
            "2. You NEVER provide your final answer until AFTER executing code multiple times\n"
            "3. Your internal planning/thinking is NOT your final answer - keep it internal\n"
            "4. You ONLY provide final answer when you have executed code, analyzed data, and have specific results\n"
            "5. Final answer MUST start with '## Executive Summary' and contain real numbers from your analysis\n"
            "\n⚠️ WRONG: Providing 'Here is my plan... Step 1: I will load data...'\n"
            "✅ RIGHT: Execute code → get results → provide report with '## Executive Summary' and actual data"
        ),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,  # Sufficient iterations for comprehensive multi-step analysis with visualizations
        memory=False,  # Disable memory to keep each query independent
        respect_context_window=True,
        step_callback=None  # Ensure clean final output
    )

    return agent


def create_deep_research_task(agent: Agent, user_question: str) -> Task:
    """
    Create a comprehensive deep research task.

    The task guides the agent to:
    1. Understand and decompose the problem
    2. Plan a multi-step approach
    3. Execute each step with code
    4. Synthesize findings into a comprehensive report
    """
    dataset_context = get_dataset_context()

    task = Task(
        description=(
            f"{dataset_context}\n\n"
            f"RESEARCH QUESTION: {user_question}\n\n"
            f"⚠️ CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
            f"1. You MUST use the execute_python_code() tool multiple times to perform analysis\n"
            f"2. Do NOT just write code - EXECUTE it using the tool!\n"
            f"3. Do NOT provide your final answer until AFTER you have executed code\n"
            f"4. Your planning/workflow is NOT your final answer - it's just your internal process\n"
            f"5. ONLY provide final answer when you have:\n"
            f"   ✓ Executed Python code multiple times\n"
            f"   ✓ Analyzed the output from your code\n"
            f"   ✓ Created visualizations with save_figure()\n"
            f"   ✓ Have specific numbers and statistics to report\n\n"
            f"❌ WRONG FINAL ANSWER: 'Here's my plan... Step 1: Load data... Step 2: Calculate...'\n"
            f"✅ CORRECT FINAL ANSWER: '## Executive Summary\\nBased on analysis of DJ30 stocks...'\n\n"
            f"YOUR WORKFLOW:\n"
            f"1. PLAN: Break down the question into 3-5 analytical steps\n"
            f"2. EXECUTE: Use execute_python_code() for each step - write comprehensive code that:\n"
            f"   - Loads and preprocesses data\n"
            f"   - Calculates relevant metrics (fundamentals, momentum, risk-adjusted returns)\n"
            f"   - Generates meaningful statistics\n"
            f"   - Creates visualizations using save_figure(fig, 'descriptive_name')\n"
            f"3. ANALYZE: Review code output and identify key insights\n"
            f"4. ITERATE: Execute more code as needed for deeper analysis\n"
            f"5. SYNTHESIZE: Compile findings into comprehensive markdown report\n\n"
            f"VISUALIZATION GUIDANCE:\n"
            f"- Create multiple visualizations (aim for 3-5) to tell a complete story\n"
            f"- Include: scatter plots (relationships), bar charts (comparisons), time series (trends)\n"
            f"- Use save_figure() inside your Python code to save each visualization\n"
            f"- Always print the returned Visualization ID\n\n"
            f"CODE EXECUTION GUIDELINES:\n"
            f"- Write comprehensive, well-documented code\n"
            f"- Handle missing data gracefully (fillna, dropna)\n"
            f"- Print key statistics and findings (use .head(), .describe(), summaries)\n"
            f"- Create composite scores by combining multiple metrics\n"
            f"- Normalize/rank metrics appropriately for comparisons\n"
            f"- DO NOT save pickle files - use in-memory processing only\n\n"
            f"OUTPUT REQUIREMENTS:\n"
            f"- Print concise summaries (avoid full dataframes)\n"
            f"- Capture visualization IDs and reference them in your final report\n"
            f"- Provide specific numbers, percentages, and statistics\n"
            f"- Explain WHY stocks are recommended (not just WHAT stocks)\n"
        ),
        agent=agent,
        expected_output=(
            "🚨 CRITICAL - FINAL ANSWER REQUIREMENTS:\n\n"
            "You may NOT provide your final answer until you have:\n"
            "1. ✅ Used execute_python_code() tool at least 3 times\n"
            "2. ✅ Created at least 2 visualizations with save_figure()\n"
            "3. ✅ Have specific numbers, percentages, and statistics from your code output\n\n"
            "❌ DO NOT return as final answer:\n"
            "- Your planning or workflow ('Here's my approach...', 'Step 1: Load data...')\n"
            "- Code snippets that weren't executed (```python code blocks)\n"
            "- Generic explanations without specific data\n"
            "- Intermediate thinking or analysis steps\n\n"
            "✅ ONLY return as final answer:\n"
            "A complete, polished analytical report that starts with '## Executive Summary'\n\n"
            "Your final answer MUST be a comprehensive analytical report in markdown format with:\n\n"
            "## Executive Summary\n"
            "2-3 sentences summarizing key findings and recommendations\n\n"
            "## Methodology\n"
            "Describe your analytical approach:\n"
            "- What metrics you calculated (fundamentals, momentum, Sharpe ratios, etc.)\n"
            "- How you combined/weighted different factors\n"
            "- Time periods analyzed\n"
            "- Data sources used\n\n"
            "## Analysis & Findings\n"
            "Detailed results with SPECIFIC NUMBERS from your code execution:\n"
            "- For each recommended stock, provide key metrics (PE ratio, ROE, momentum %, Sharpe ratio)\n"
            "- Explain WHY each stock was selected based on the data\n"
            "- Include comparative statistics (e.g., 'Stock A has 23% higher ROE than average')\n\n"
            "## Key Insights\n"
            "Bullet points highlighting:\n"
            "- Most important discoveries from your analysis\n"
            "- Patterns or trends identified\n"
            "- Notable outliers or surprises\n\n"
            "## Visualizations\n"
            "Reference ALL created charts with their Visualization IDs:\n"
            "- Format: 'Visualization ID: viz_20251118_123456_abc123'\n"
            "- Briefly describe what each visualization shows\n\n"
            "## Conclusions\n"
            "Summary and actionable implications of your findings\n\n"
            "FORMATTING NOTES:\n"
            "- Use proper markdown headers (##, ###) for sections\n"
            "- Use bullet points for lists\n"
            "- Bold key terms and numbers for emphasis\n"
            "- Do NOT include internal 'Thought:', 'Action:', or tool-calling syntax in final report\n"
            "- Write as a polished, professional analysis report"
        )
    )

    return task


def run_deep_research(user_question: str) -> str:
    """
    Execute a deep research analysis with detailed logging.

    Args:
        user_question: The complex question requiring deep analysis

    Returns:
        str: Comprehensive research report
    """
    import logging
    import traceback

    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        print("\n" + "=" * 80)
        print("🔬 DEEP RESEARCH MODE ACTIVATED")
        print("=" * 80)
        print(f"📝 Question: {user_question}")
        print(f"📁 Working Directory: {os.getcwd()}")
        print(f"🤖 LLM Provider: {os.getenv('SELECTED_LLM', 'gemini')}")

        # Check LLM configuration
        selected_llm = os.getenv("SELECTED_LLM", "gemini")
        if selected_llm == "gemini":
            model_id = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash-exp")
        elif selected_llm == "openai":
            model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4")
        else:
            model_id = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash-exp")
        print(f"🧠 Model: {model_id}")

        # Check data files
        data_dir = './data'
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            print(f"📊 Found {len(csv_files)} CSV files in {data_dir}/")
        else:
            print(f"⚠️  Warning: Data directory '{data_dir}' not found")

        print("\n" + "-" * 80)
        print("🚀 STEP 1: Creating Deep Research Agent...")
        print("-" * 80)
        agent = create_deep_research_agent()
        print("✅ Agent created successfully")
        print(f"   - Role: {agent.role}")
        print(f"   - Tools: {len(agent.tools)} tool(s)")
        print(f"   - Max Iterations: {agent.max_iter}")

        print("\n" + "-" * 80)
        print("🚀 STEP 2: Creating Research Task...")
        print("-" * 80)
        task = create_deep_research_task(agent, user_question)
        print("✅ Task created successfully")
        print(f"   - Description length: {len(task.description)} chars")
        print(f"   - Expected output defined: Yes")

        print("\n" + "-" * 80)
        print("🚀 STEP 3: Initializing Crew and Starting Execution...")
        print("-" * 80)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        print("✅ Crew initialized")
        print("\n🔄 Executing research... (this may take several minutes)\n")

        result = crew.kickoff()

        print("\n" + "-" * 80)
        print("✅ EXECUTION COMPLETE")
        print("-" * 80)
        print(f"📄 Result length: {len(str(result))} characters")
        print(f"📊 Result type: {type(result)}")

        # Clean up the result to remove any leaked internal agent syntax
        result_str = str(result)

        # Remove common agent internal syntax patterns
        import re

        # Strategy: Find the first markdown heading (## Something) and return everything from there
        # This is more robust than trying to remove all thought patterns
        match = re.search(r'^##\s+', result_str, flags=re.MULTILINE)

        if match:
            # Found the start of the actual report
            cleaned_result = result_str[match.start():]
            print("🧹 Extracted clean report (removed internal agent thoughts)")
        else:
            # Fallback: Try to remove known patterns
            print("⚠️  No markdown heading found, using fallback cleanup")

            # Remove everything before the first paragraph that doesn't contain agent keywords
            # Split by double newlines to get paragraphs
            paragraphs = result_str.split('\n\n')

            # Skip paragraphs that contain agent reasoning keywords
            skip_keywords = ['Thought:', 'Action:', 'Action Input:', 'Observation:',
                           'I need to', 'I will', 'I have now', 'I can now',
                           'Looking at the', 'I expect', 'I will analyze']

            clean_paragraphs = []
            found_content = False

            for para in paragraphs:
                # Once we find content without agent keywords, include everything from there
                if not found_content:
                    # Check if this paragraph contains agent reasoning
                    has_agent_keywords = any(keyword in para for keyword in skip_keywords)
                    if not has_agent_keywords and len(para.strip()) > 50:
                        found_content = True
                        clean_paragraphs.append(para)
                else:
                    clean_paragraphs.append(para)

            cleaned_result = '\n\n'.join(clean_paragraphs)

        # Additional cleanup: Remove any remaining single-line agent syntax
        cleaned_result = re.sub(r'^(Thought|Action|Action Input|Observation):.*$', '', cleaned_result, flags=re.MULTILINE)

        # Clean up excessive whitespace
        cleaned_result = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_result)

        # Strip leading/trailing whitespace
        cleaned_result = cleaned_result.strip()

        print("=" * 80 + "\n")

        return cleaned_result

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR DURING DEEP RESEARCH")
        print("=" * 80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")

        # Check for specific error types
        error_str = str(e).lower()
        if "keys must be enclosed" in error_str or "json" in error_str:
            print("\n⚠️  JSON PARSING ERROR DETECTED")
            print("-" * 80)
            print("This usually happens when:")
            print("1. The CodeInterpreterTool returns malformed output")
            print("2. The agent's response contains unescaped quotes or special characters")
            print("3. Context window is exceeded, causing truncated responses")
            print("\n🔧 Suggested fixes:")
            print("- Try a simpler question first to test the setup")
            print("- Check if crewai-tools is properly installed: pip install --upgrade crewai-tools")
            print("- Reduce max_iter in the agent configuration")
            print("- Try using a different LLM model")

        print("\n📋 Full Traceback:")
        print("-" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")

        # Return a user-friendly error message
        if "keys must be enclosed" in error_str or "json" in error_str:
            return (
                "⚠️ Deep Research encountered a JSON parsing error.\n\n"
                "This is likely due to an issue with the CodeInterpreterTool or context overflow. "
                "Please try:\n"
                "1. A simpler question to test the setup\n"
                "2. Upgrading crewai-tools: `pip install --upgrade crewai-tools`\n"
                "3. Using Standard Mode instead of Deep Research\n\n"
                f"Technical details: {type(e).__name__}: {str(e)}"
            )
        else:
            return f"Error during deep research: {type(e).__name__}: {str(e)}\n\nPlease check the logs above for details."

