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
        # Sanitize model ID for LiteLLM
        if model_id.startswith("models/"):
            model_id = model_id.replace("models/", "")
        if not model_id.startswith("gemini/"):
            model_id = f"gemini/{model_id}"

    elif selected_llm == "openai":
        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4")
    else:
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash-exp")
        if model_id.startswith("models/"):
            model_id = model_id.replace("models/", "")
        if not model_id.startswith("gemini/"):
            model_id = f"gemini/{model_id}"

    llm = LLM(model=model_id)

    # Use custom code execution tools (more reliable than CodeInterpreterTool)
    tools = [execute_python_code, save_plotly_figure]
    print("✅ Custom code execution tools initialized")

    agent = Agent(
        role="Lead Quantitative Developer",
        goal=(
            "Execute the provided analytical plan by writing and running Python code. "
            "Calculate metrics, generate visualizations, and report findings based on the Strategist's plan."
        ),
        backstory=(
            "You are an expert Python Developer and Quantitative Analyst. "
            "You work in a team with a Senior Strategist who provides you with a high-level plan. "
            "Your job is to TRANSLATE that plan into working Python code using pandas, numpy, and plotly. "
            "You are meticulous about data quality, error handling, and creating clear visualizations. "
            "You NEVER deviate from the plan without good reason, but you fix bugs autonomously."
            "\n\n🚨 CRITICAL BEHAVIORAL RULES:\n"
            "1. You ALWAYS use execute_python_code() tool to perform analysis\n"
            "2. You follow the steps provided in the context/plan\n"
            "3. You ONLY provide final answer when you have executed code and have results\n"
            "4. Final answer MUST start with '## Executive Summary' and contain real numbers\n"
            "5. You NEVER write code blocks (```python) in your text response - ONLY inside the tool"
        ),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=25,
        memory=False,
        respect_context_window=True,
        step_callback=None
    )

    return agent


def create_deep_research_task(agent: Agent, user_question: str, plan_context: str = None) -> Task:
    """
    Create a task to execute the research plan.
    """
    dataset_context = get_dataset_context()

    # If plan is provided (from Planner Agent), include it. Otherwise fallback to self-planning.
    plan_instruction = ""
    if plan_context:
        plan_instruction = f"✅ APPROVED PLAN FROM STRATEGIST:\n{plan_context}\n\nEXECUTE THIS PLAN STEP-BY-STEP."
    else:
        plan_instruction = "Develop and execute a plan to answer the question."

    task = Task(
        description=(
            f"{dataset_context}\n\n"
            f"RESEARCH QUESTION: {user_question}\n\n"
            f"{plan_instruction}\n\n"
            f"⚠️ CRITICAL INSTRUCTIONS:\n"
            f"1. You MUST use the execute_python_code() tool multiple times\n"
            f"2. Do NOT just write code in markdown - EXECUTE it!\n"
            f"3. Do NOT provide your final answer until AFTER you have executed code\n"
            f"4. ONLY provide final answer when you have specific numbers and charts\n\n"
            f"YOUR WORKFLOW:\n"
            f"1. REVIEW the provided plan\n"
            f"2. EXECUTE each step using execute_python_code()\n"
            f"   - Load data, calculate metrics, generate stats\n"
            f"   - Use get_news_sentiment() if required by plan\n"
            f"   - Create visualizations using save_figure()\n"
            f"3. ANALYZE output and synthesize findings\n\n"
            f"VISUALIZATION GUIDANCE:\n"
            f"- Create multiple visualizations (aim for 3-5)\n"
            f"- Use save_figure() inside your Python code\n"
            f"- Always print the returned Visualization ID\n\n"
            f"OUTPUT REQUIREMENTS:\n"
            f"- Print concise summaries (avoid full dataframes)\n"
            f"- Capture visualization IDs\n"
            f"- Provide specific numbers and statistics\n"
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
            "- When quoting companies, use their names instead of tickers\n"
            "- Bold key terms and numbers for emphasis\n"
            "- Do NOT include internal 'Thought:', 'Action:', or tool-calling syntax in final report\n"
            "- Write as a polished, professional analysis report"
        )
    )

    return task


def generate_research_plan(user_question: str) -> str:
    """
    Generate a research plan for the given question using the Planner Agent.
    """
    import logging
    import os
    from crewai import Crew, Agent, Task, LLM

    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    print("\n" + "=" * 80)
    print("🔬 DEEP RESEARCH: PLANNING PHASE")
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
    print("🚀 STEP 1: Initializing Planner Agent...")
    print("-" * 80)

    from agents.planner_agent import create_planner_agent, create_planning_task

    # Create Planner Agent
    planner = create_planner_agent()
    print(f"✅ Planner Agent created: {planner.role}")

    print("\n" + "-" * 80)
    print("🚀 STEP 2: Planning Phase...")
    print("-" * 80)

    # Create Planning Task
    plan_task = create_planning_task(planner, user_question)

    # Run Planner Crew
    planner_crew = Crew(
        agents=[planner],
        tasks=[plan_task],
        verbose=True
    )

    print("🔄 Generating research plan...")
    planner_crew.kickoff()

    # Extract the plan
    plan_output = str(plan_task.output)
    print("\n" + "-" * 80)
    print("📋 EXTRACTED PLAN")
    print("-" * 80)
    print(plan_output[:500] + "..." if len(plan_output) > 500 else plan_output)

    return plan_output


def execute_research_plan(user_question: str, plan_output: str) -> str:
    """
    Execute a research plan using the Deep Research Analyst (Executor).
    """
    import logging
    import re
    from crewai import Crew, Agent, Task, LLM

    # Configure logging (ensure it's configured if this is called standalone)
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    print("\n" + "=" * 80)
    print("� DEEP RESEARCH: EXECUTION PHASE")
    print("=" * 80)
    print(f"📝 Question: {user_question}")
    print("\n" + "-" * 80)
    print("🚀 STEP 3: Initializing Executor Agent...")
    print("-" * 80)

    # Create Executor Agent
    executor = create_deep_research_agent()
    print(f"✅ Executor Agent created: {executor.role}")

    print("\n" + "-" * 80)
    print("🚀 STEP 4: Execution Phase...")
    print("-" * 80)

    # Create Execution Task (receives plan via string context)
    exec_task = create_deep_research_task(executor, user_question, plan_context=plan_output)

    # Run Executor Crew
    executor_crew = Crew(
        agents=[executor],
        tasks=[exec_task],
        verbose=True
    )

    print("🔄 Executing research... (this may take several minutes)\n")
    result = executor_crew.kickoff()

    print("\n" + "-" * 80)
    print("✅ EXECUTION COMPLETE")
    print("-" * 80)
    print(f"📄 Result length: {len(str(result))} characters")
    print(f"📊 Result type: {type(result)}")

    # Clean up the result
    result_str = str(result)

    # Strategy: Find the first markdown heading (## Something) and return everything from there
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

    return cleaned_result
