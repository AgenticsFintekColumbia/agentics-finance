from crewai import Agent, Task, Crew, LLM
import os

# Initialize LLM
model_id = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-1.5-flash")

# Sanitize model ID for LiteLLM
# LiteLLM expects 'gemini/model-name'
# If user provides 'models/gemini/model-name', we strip 'models/'
if model_id.startswith("models/"):
    model_id = model_id.replace("models/", "")

# Ensure it starts with 'gemini/' if it's a gemini model
if "gemini" in model_id and not model_id.startswith("gemini/"):
    model_id = f"gemini/{model_id}"

llm = LLM(
    model=model_id,
    verbose=True,
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

def classify_query(query: str) -> str:
    """
    Classifies a financial query as either 'simple' or 'complex'.

    Simple: Can be answered with direct data lookup (price, PE ratio, basic chart).
    Complex: Requires multi-step reasoning, code execution, simulation, or cross-domain analysis.

    Returns: 'simple' or 'complex'
    """
    classification_agent = Agent(
        role="Query Classifier",
        goal="Classify financial queries as 'simple' or 'complex' based on required analysis depth.",
        backstory="You are an expert system architect who routes queries to the right specialist. "
                  "Simple queries go to the Data Analyst (lookup/basic stats). "
                  "Complex queries go to the Deep Research Analyst (code/simulation/multi-step).",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    task = Task(
        description=f"""
        Classify the following query into exactly one category: 'simple' or 'complex'.

        QUERY: "{query}"

        GUIDELINES:
        - 'simple': "What is Apple's PE?", "Show me the price of BTC", "Plot VIX last month".
          (Keywords: What is, Show, Plot, Price, Ratio, Basic stats)

        - 'complex': "Simulate a strategy...", "Find correlation between...", "Optimize portfolio...", "Why did...", "Compare volatility regimes...".
          (Keywords: Simulate, Optimize, Correlate, Why, Explain, Backtest, Strategy, Regimes)

        OUTPUT FORMAT:
        Just the word 'simple' or 'complex'. No other text.
        """,
        agent=classification_agent,
        expected_output="A single word: 'simple' or 'complex'"
    )

    crew = Crew(
        agents=[classification_agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()
    return str(result).strip().lower()
