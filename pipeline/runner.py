from agentics import AG
from crewai import Agent, Task, Crew
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from .transduction_pipeline import TransductionPipeline
# Load environment variables
load_dotenv()

_selected_columns_from_ui: Optional[List[str]] = None
_date_range_from_ui: Optional[Tuple[str, str]] = None


def set_selected_columns(columns: Optional[List[str]]):
    """Set the selected columns from the UI. Called by Streamlit app before running analysis."""
    global _selected_columns_from_ui
    _selected_columns_from_ui = columns


def get_selected_columns() -> Optional[List[str]]:
    """Get the selected columns from the UI. Called by the tool during execution."""
    return _selected_columns_from_ui


def set_date_range(start_date: str, end_date: str):
    """Set the date range from the UI. Called by Streamlit app before running analysis."""
    global _date_range_from_ui
    _date_range_from_ui = (start_date, end_date)


def get_date_range() -> Optional[Tuple[str, str]]:
    """Get the date range from the UI. Called by the tool during execution."""
    return _date_range_from_ui


def run_analysis(user_question: str, selected_columns: list = None) -> str:
    # Set selected columns in the tool module so the tool can read it deterministically
    set_selected_columns(selected_columns)
    """
    Run a complete analysis for a user question with conversation context.

    Args:
        user_question: The user's question
        conversation_history: List of previous messages for context
        selected_columns: List of column names to include in transduction analysis

    Returns:
        str: The agent's analysis and response, formatted for display
    """
    try:
        pipeline = TransductionPipeline()
        result = pipeline.run(user_question)

        # Check if the analysis was successful
        if result.get('success'):
            # Format the response with both detailed answer and explanation
            detailed_answer = result.get('detailed_answer', 'No answer generated')
            explanation = result.get('explanation', '')
            date_range = result.get('date_range', {})

            # Build formatted response
            formatted_response = f"\n\n### Analysis\n\n"
            formatted_response += f"{detailed_answer}"

            # Add explanation if available
            if explanation:
                formatted_response += f"\n\n### Methodology\n\n{explanation}"

            # Add metadata footer
            if date_range:
                rows_analyzed = date_range.get('rows_analyzed', 0)
                total_rows = date_range.get('total_rows_in_range', 0)
                sampling_ratio = date_range.get('sampling_ratio', '1:1')
                num_batches = date_range.get('num_batches', 0)

                formatted_response += f"\n\n---\n\n"
                formatted_response += f"**Analysis Details:**\n"
                formatted_response += f"- Date Range: {date_range.get('start')} to {date_range.get('end')}\n"
                formatted_response += f"- Data Points: {rows_analyzed:,} rows analyzed"
                if sampling_ratio != '1:1':
                    formatted_response += f" (sampled {sampling_ratio} from {total_rows:,} total rows)"
                formatted_response += f"\n- Processing: {num_batches} batches"

            return formatted_response
        else:
            return f"❌ **Error:** {result.get('error', 'Unknown error occurred')}"

    except Exception as e:
        return f"❌ **Error during analysis:** {str(e)}"
