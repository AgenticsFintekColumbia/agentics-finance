"""CrewAI agents for financial analysis."""

# Import all functions from financial_analyst module
from .financial_analyst import (
    create_financial_analyst_agent,
    create_analysis_task,
    run_analysis,
    get_tool_categories
)

# Import deep research functions
from .deep_research_analyst import (
    run_deep_research,
    create_deep_research_agent,
    create_deep_research_task
)

__all__ = [
    'create_financial_analyst_agent',
    'create_analysis_task',
    'run_analysis',
    'get_tool_categories',
    'run_deep_research',
    'create_deep_research_agent',
    'create_deep_research_task'
]

