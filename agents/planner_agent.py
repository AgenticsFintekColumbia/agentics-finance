from crewai import Agent, Task, LLM
import os
from dotenv import load_dotenv

load_dotenv()

def create_planner_agent():
    """
    Create a Senior Financial Strategist (Planner) agent.

    Role: Architect/Planner
    Goal: Decompose complex queries into precise analytical steps.
    Tools: None (Pure reasoning).
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

    agent = Agent(
        role="Senior Financial Strategist",
        goal="Decompose complex financial questions into a precise, step-by-step analytical plan for a quantitative developer.",
        backstory=(
            "You are a Senior Financial Strategist with decades of experience in investment banking and quantitative research. "
            "Your job is NOT to write code, but to DESIGN the analytical strategy. "
            "You receive a complex question (e.g., 'Optimize a portfolio for Sharpe ratio') and break it down into "
            "logical, executable steps for your Quantitative Developer Agent (who will write the Python code). "
            "You must specify exactly what data to load, what metrics to calculate, and what visualizations to create."
        ),
        tools=[], # No tools, pure reasoning
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=False
    )

    return agent

def create_planning_task(agent: Agent, user_question: str) -> Task:
    """
    Create a task for the Planner Agent.
    """
    return Task(
        description=f"""
        ANALYZE the following financial research question and CREATE a detailed step-by-step plan.

        RESEARCH QUESTION: "{user_question}"

        Your plan must be designed for a Python Developer who has access to:
        1. Macroeconomic data (Fed funds, CPI, Unemployment)
        2. Market Factors (S&P 500, VIX, Treasury Yields, Commodities)
        3. Company Fundamentals (EPS, ROE, PE, Growth)
        4. DJ30 Stock Prices (OHLCV)
        5. News Sentiment (via get_news_sentiment tool)

        REQUIREMENTS:
        - Break the analysis into 3-5 distinct, logical steps.
        - For each step, specify:
            - What data to load/filter.
            - What calculations to perform (be specific: 'Calculate 12-month rolling volatility').
            - What visualizations to generate (e.g., 'Scatter plot of Rate vs Tech Sector').
        - The final step must be 'Synthesis and Reporting'.

        OUTPUT FORMAT:
        Return a clear, numbered list of steps. Do not write code.
        """,
        agent=agent,
        expected_output="A numbered list of 3-5 analytical steps, describing the data, calculations, and visualizations required."
    )
