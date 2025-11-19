"""
Streamlit application for financial data Q&A with CrewAI agents.
"""

import streamlit as st
import os
import re
import sys
from io import StringIO
import pandas as pd
from pipeline import run_analysis
from utils import get_data_summary, get_column_descriptions, get_firm_data_summary, get_firm_column_descriptions, get_dj30_data_summary, get_dj30_column_descriptions

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Transduction Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: var(--secondary-background-color);
        border-left-color: #1f77b4;
        color: var(--text-color);
    }
    .assistant-message {
        background-color: var(--secondary-background-color);
        border-left-color: #2ca02c;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: var(--text-color);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1557a0;
        border: none;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }

    /* Allow sidebar to be resized larger (up to 60% of page width) */
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 60% !important;
    }

    /* Ensure sidebar content is scrollable when extended */
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto;
    }

    /* Style for example question buttons */
    .example-question-button {
        text-align: left;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
        color: #333;
        font-size: 0.9rem;
        transition: all 0.2s;
    }

    .example-question-button:hover {
        background-color: #e7f3ff;
        border-color: #1f77b4;
        transform: translateX(4px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = ""

if "show_logs" not in st.session_state:
    st.session_state.show_logs = True

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "selected_transduction_columns" not in st.session_state:
    st.session_state.selected_transduction_columns = []

if "transduction_flow" not in st.session_state:
    st.session_state.transduction_flow = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"

# Initialize date range for transduction analysis
if "transduction_start_date" not in st.session_state:
    from datetime import date
    st.session_state.transduction_start_date = date(2018, 1, 1)

if "transduction_end_date" not in st.session_state:
    from datetime import date
    st.session_state.transduction_end_date = date(2025, 1, 1)


def get_merged_data_columns():
    """Get all column names from merged_data.csv (or split files)."""
    from utils.csv_reader import read_merged_data_header

    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        header = read_merged_data_header(data_dir)
        return header
    except Exception as e:
        st.error(f"Error reading column names: {e}")
        return []


def escape_markdown_special_chars(text: str) -> str:
    """
    Escape special characters that cause markdown rendering issues.
    This prevents dollar signs from being interpreted as LaTeX math delimiters.
    """
    # Escape dollar signs to prevent LaTeX interpretation
    # Use a zero-width space or backslash to break the pattern
    text = text.replace('$', r'\$')
    return text


def run_analysis_with_logs(user_input: str, selected_columns: list = None, start_date: str = None, end_date: str = None) -> str:
    """Run analysis and capture stdout/stderr to display in logs."""
    # Set selected columns and date range in the tool module before running analysis
    # This ensures the tool reads the deterministic UI selection, not agent-provided values
    from pipeline.runner import set_selected_columns, set_date_range
    set_selected_columns(selected_columns)
    if start_date and end_date:
        set_date_range(start_date, end_date)

    # Create a StringIO object to capture output
    captured_output = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect stdout and stderr to capture output
        sys.stdout = captured_output
        sys.stderr = captured_output

        # Run the analysis (selected columns are already set in tool module)
        response = run_analysis(user_input, selected_columns)

        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Get captured output and clean ANSI escape codes
        raw_logs = captured_output.getvalue()
        # Remove ANSI color codes (e.g., [36m, [0m, [1;36m, etc.)
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        cleaned_logs = ansi_escape.sub('', raw_logs)

        # Store the cleaned logs in session state
        st.session_state.agent_logs = cleaned_logs

        return response

    except Exception as e:
        # Restore original stdout/stderr in case of error
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Clean and store what we captured plus the error
        raw_logs = captured_output.getvalue()
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        cleaned_logs = ansi_escape.sub('', raw_logs)
        st.session_state.agent_logs = cleaned_logs + f"\n\nError: {str(e)}"
        raise e


def get_columns_for_question(question: str, all_columns: list) -> list:
    """
    Manually map questions to their required columns.
    Returns a list of exact column names that should be pre-selected.
    """
    # Define question-to-columns mapping
    question_mappings = {
        # Question 1: Explain how AMZN and AAPL's strategy shifted over time, and any major investments they made from 2020 onwards
        "explain how amzn and aapl's strategy shifted over time, and any major investments they made from 2020 onwards": [
            # News and earnings for strategy/investment information
            'news_AAPL', 'news_AMZN', 'Earningcall_AAPL', 'Earningcall_AMZN', 'Headlines',
            # Price data to track performance
            'open_AAPL', 'high_AAPL', 'low_AAPL', 'close_AAPL', 'adj_close_AAPL', 'volume_AAPL',
            # Fundamental metrics for strategy analysis
            'EPS_AAPL_ACTUAL', 'EPS_AAPL_MEDEST', 'EPS_AMZN_ACTUAL', 'EPS_AMZN_MEDEST',
            'ROE_AAPL_ACTUAL', 'ROE_AMZN_ACTUAL',
            'SAL_AAPL_ACTUAL', 'SAL_AMZN_ACTUAL',
            'NET_AAPL_ACTUAL', 'NET_AMZN_ACTUAL'
        ],

        # Question 2: Why did the stock price of AAPL drop in March 2020?
        "why did the stock price of aapl drop in march 2020?": [
            # Price data to see the drop
            'open_AAPL', 'high_AAPL', 'low_AAPL', 'close_AAPL', 'adj_close_AAPL', 'volume_AAPL',
            # News to explain why
            'news_AAPL', 'news_market', 'Headlines',
            # Earnings calls around that time
            'Earningcall_AAPL',
            # Market context (use original CSV column names - tool will sanitize them)
            '^GSPC', '^VIX'
        ],

        # Question 3: What were the key factors behind NVDA's stock price surge in 2023?
        # Note: NVDA doesn't have price columns in this dataset, so we focus on news, earnings, and fundamentals
        "what were the key factors behind nvda's stock price surge in 2023?": [
            # News and earnings
            'news_NVDA', 'Earningcall_NVDA', 'Headlines',
            # Fundamentals
            'EPS_NVDA_ACTUAL', 'EPS_NVDA_MEDEST',
            'ROE_NVDA_ACTUAL',
            'SAL_NVDA_ACTUAL', 'NET_NVDA_ACTUAL',
            # Market context (use original CSV column names - tool will sanitize them)
            '^GSPC', '^VIX'
        ],

        # Question 4: Analyze the relationship between MSFT's earnings announcements and its stock price movements from 2020-2023
        "analyze the relationship between msft's earnings announcements and its stock price movements from 2020-2023": [
            # Price data
            'open_MSFT', 'high_MSFT', 'low_MSFT', 'close_MSFT', 'adj_close_MSFT', 'volume_MSFT',
            # Earnings calls
            'Earningcall_MSFT',
            # News
            'news_MSFT', 'Headlines',
            # Earnings fundamentals
            'EPS_MSFT_ACTUAL', 'EPS_MSFT_MEDEST',
            'ROE_MSFT_ACTUAL',
            'NET_MSFT_ACTUAL'
        ],

        # Question 5: What market events and company-specific news drove JPM's volatility during the 2022-2023 period?
        "what market events and company-specific news drove jpm's volatility during the 2022-2023 period?": [
            # Price data for volatility
            'open_JPM', 'high_JPM', 'low_JPM', 'close_JPM', 'adj_close_JPM', 'volume_JPM',
            # News
            'news_JPM', 'news_market', 'Headlines',
            # Earnings
            'Earningcall_JPM',
            # Market context (use original CSV column names - tool will sanitize them)
            '^GSPC', '^VIX', 'DGS10',
            # Fundamentals (JPM uses returnonequity instead of ROE)
            'returnonequity_JPM'
        ]
    }

    # Normalize question for lookup
    question_normalized = question.lower().strip()

    # Check for exact match
    if question_normalized in question_mappings:
        selected_cols = question_mappings[question_normalized]
        # Filter to only include columns that actually exist in the dataset
        return [col for col in selected_cols if col in all_columns]

    # If no match found, return empty list
    return []


def set_example_question(question: str):
    """Set the example question in the text input and pre-select relevant columns."""
    # Set the question text
    st.session_state.user_input = question

    # Get all columns and determine which ones to select
    all_columns = get_merged_data_columns()
    if all_columns:
        relevant_cols = get_columns_for_question(question, all_columns)
        # Remove duplicates and ensure we have a clean list
        relevant_cols = list(set(relevant_cols))
        # Set session state - this will be reflected in checkboxes on next render
        st.session_state.selected_transduction_columns = relevant_cols

        # Clear all existing checkbox states so they can be reinitialized from selected_transduction_columns
        # This prevents the error of modifying widget state after instantiation
        all_columns_list = get_merged_data_columns()
        if all_columns_list:
            for col in all_columns_list:
                checkbox_key = f"col_{col}"
                # Delete existing checkbox state - it will be reinitialized on next render
                if checkbox_key in st.session_state:
                    del st.session_state[checkbox_key]

        print(f"🔧 Set {len(relevant_cols)} columns for question: {question[:50]}...")
        if relevant_cols:
            print(f"   Selected columns: {relevant_cols}")

    # Rerun to update the UI with new selections
    st.rerun()


def display_transduction_flow():
    """Display the transduction flow visualization."""
    st.markdown('<div class="main-header">Transduction Flow</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualize how data flows through the transduction process</div>', unsafe_allow_html=True)

    if not st.session_state.transduction_flow:
        st.info("💡 No transduction flow data available. Run a transduction analysis in the Chat page to see the flow here.")
        return

    flow = st.session_state.transduction_flow

    # Helper function to get Pydantic model definition
    def get_model_definition(model_class):
        """Get the Pydantic model definition as a string."""
        try:
            import inspect
            from typing import Union, get_origin, get_args
            # Try to get source code
            try:
                source = inspect.getsource(model_class)
                return source
            except (OSError, TypeError):
                # If source not available, reconstruct from model_fields
                if hasattr(model_class, 'model_fields'):
                    lines = [f"class {model_class.__name__}(BaseModel):"]
                    for field_name, field_info in model_class.model_fields.items():
                        # Get field type annotation
                        field_type = field_info.annotation
                        field_type_str = None

                        # Handle Union types (including Optional)
                        origin = get_origin(field_type)
                        if origin is Union:
                            args = get_args(field_type)
                            # Check if it's Optional (Union with None)
                            non_none_args = [arg for arg in args if arg is not type(None)]
                            if len(non_none_args) == 1 and len(args) == 2:
                                # It's Optional[T]
                                type_name = getattr(non_none_args[0], '__name__', str(non_none_args[0]))
                                field_type_str = f"{type_name} | None"
                            else:
                                # It's a Union of multiple types
                                type_names = [getattr(arg, '__name__', str(arg)) for arg in non_none_args]
                                field_type_str = " | ".join(type_names)
                        else:
                            # Regular type
                            field_type_str = getattr(field_type, '__name__', str(field_type))

                        # Get default value
                        default = field_info.default
                        if default is None:
                            default_str = "None"
                        elif isinstance(default, str):
                            default_str = f'"{default}"'
                        else:
                            default_str = str(default)

                        # Get description if available
                        description = field_info.description
                        if description:
                            # Escape quotes in description
                            description_escaped = description.replace('"', '\\"')
                            lines.append(f"    {field_name}: {field_type_str} = Field(")
                            lines.append(f"        {default_str},")
                            lines.append(f'        description="{description_escaped}"')
                            lines.append(f"    )")
                        else:
                            lines.append(f"    {field_name}: {field_type_str} = {default_str}")

                    return "\n".join(lines)
                else:
                    return f"class {model_class.__name__}(BaseModel):\n    # Model definition not available"
        except Exception as e:
            return f"# Error getting model definition: {e}"

    # Helper function to convert AG to dataframe
    def ag_to_df(ag_obj):
        """Convert an AG object to a pandas DataFrame."""
        try:
            if hasattr(ag_obj, 'to_dataframe'):
                return ag_obj.to_dataframe()
            elif hasattr(ag_obj, 'states'):
                # Convert states directly to dataframe
                data = [state.model_dump() for state in ag_obj.states]
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error converting to dataframe: {e}")
            return pd.DataFrame()

    # Helper function to convert batch results to dataframe
    def batches_to_df(batches):
        """Convert a list of batch results to a pandas DataFrame."""
        try:
            data = []
            for batch in batches:
                if hasattr(batch, 'model_dump'):
                    data.append(batch.model_dump())
                elif isinstance(batch, dict):
                    data.append(batch)
                else:
                    # Try to convert to dict
                    data.append({"result": str(batch)})
            return pd.DataFrame(data)
        except Exception as e:
            st.warning(f"Error converting batches to dataframe: {e}")
            return pd.DataFrame()

    # 1. Initial States
    if "initial_states" in flow:
        st.markdown("---")
        st.markdown(f"### 1️⃣ Initial States")
        st.markdown(f"**Pydantic Class:** `{flow['initial_states']['atype_name']}`")

        # Display model definition
        if hasattr(flow['initial_states']['agentics'], 'atype') and flow['initial_states']['agentics'].atype:
            model_def = get_model_definition(flow['initial_states']['agentics'].atype)
            with st.expander("📋 View Model Definition", expanded=False):
                st.code(model_def, language="python")

        st.markdown(f"**Number of Rows:** {flow['initial_states']['num_rows']}")

        initial_df = ag_to_df(flow['initial_states']['agentics'])
        if not initial_df.empty:
            st.dataframe(initial_df, use_container_width=True, height=800)
        else:
            st.warning("Could not convert initial states to dataframe")

    # 2. Final Intermediate Result
    if "final_intermediate" in flow:
        st.markdown("---")
        st.markdown(f"### 2️⃣ Final Intermediate Result")
        st.markdown(f"**Pydantic Class:** `{flow['final_intermediate']['atype_name']}`")

        # Display model definition
        if hasattr(flow['final_intermediate']['agentics'], 'atype') and flow['final_intermediate']['agentics'].atype:
            model_def = get_model_definition(flow['final_intermediate']['agentics'].atype)
            with st.expander("📋 View Model Definition", expanded=False):
                st.code(model_def, language="python")

        st.markdown(f"**Number of Rows:** {flow['final_intermediate']['num_rows']}")

        final_intermediate_df = ag_to_df(flow['final_intermediate']['agentics'])
        if not final_intermediate_df.empty:
            st.dataframe(final_intermediate_df, use_container_width=True, height=400)
        else:
            st.warning("Could not convert final intermediate result to dataframe")

    # 3. Final Answer
    if "final_answer" in flow:
        st.markdown("---")
        st.markdown(f"### 3️⃣ Final Answer")
        st.markdown(f"**Pydantic Class:** `{flow['final_answer']['atype_name']}`")

        # Display model definition
        if hasattr(flow['final_answer']['agentics'], 'atype') and flow['final_answer']['agentics'].atype:
            model_def = get_model_definition(flow['final_answer']['agentics'].atype)
            with st.expander("📋 View Model Definition", expanded=False):
                st.code(model_def, language="python")

        st.markdown(f"**Number of Rows:** {flow['final_answer']['num_rows']}")

        final_answer_df = ag_to_df(flow['final_answer']['agentics'])
        if not final_answer_df.empty:
            st.dataframe(final_answer_df, use_container_width=True, height=50)
        else:
            st.warning("Could not convert final answer to dataframe")

    st.markdown("---")
    st.caption("💡 This flow shows how your data is transformed through the transduction process. Each stage reduces the data while preserving key insights.")


# Sidebar
with st.sidebar:
    st.markdown("#### 📁 Available Data")

    with st.expander("📈 Dataset Information"):
        data_summary = get_data_summary()
        firm_summary = get_firm_data_summary()
        dj30_summary = get_dj30_data_summary()

        st.markdown("**Macro Factors**")
        st.markdown(f"- **Date Range:** {data_summary['macro_factors']['date_range']['start']} to {data_summary['macro_factors']['date_range']['end']}")
        st.markdown(f"- **Records:** {data_summary['macro_factors']['rows']:,}")
        st.markdown(f"- **Indicators:** {len(data_summary['macro_factors']['columns'])}")

        st.markdown("**Market Factors**")
        st.markdown(f"- **Date Range:** {data_summary['market_factors']['date_range']['start']} to {data_summary['market_factors']['date_range']['end']}")
        st.markdown(f"- **Records:** {data_summary['market_factors']['rows']:,}")
        st.markdown(f"- **Indicators:** {len(data_summary['market_factors']['columns'])}")

        st.markdown("**Company Fundamentals**")
        st.markdown(f"- **Date Range:** {firm_summary['date_range']['start']} to {firm_summary['date_range']['end']}")
        st.markdown(f"- **Records:** {firm_summary['total_records']:,}")
        st.markdown(f"- **Companies:** {firm_summary['unique_tickers']}")
        st.markdown(f"- **Metrics:** EPS, ROE, ROA, P/E, Margins, Growth")

        st.markdown(dj30_summary)

    with st.expander("📋 Available Indicators"):
        descriptions = get_column_descriptions()
        firm_descriptions = get_firm_column_descriptions()
        dj30_descriptions = get_dj30_column_descriptions()

        st.markdown("**Macroeconomic Indicators:**")
        for indicator, desc in descriptions["macro_factors"].items():
            st.markdown(f"- **{indicator}**: {desc}")

        st.markdown("\n**Market Indicators:**")
        for indicator, desc in descriptions["market_factors"].items():
            if indicator != "Headlines":
                st.markdown(f"- **{indicator}**: {desc}")

        st.markdown("\n**Company Fundamental Metrics:**")
        # Show key metrics only (not all the forward growth/volatility variants)
        key_metrics = ["TICKER", "STATPERS", "PRICE", "EBS", "EPS", "DPS", "ROA", "ROE", "NAV", "GRM"]
        for metric in key_metrics:
            if metric in firm_descriptions:
                st.markdown(f"- **{metric}**: {firm_descriptions[metric]}")
        st.markdown("- Plus forward 1-year growth and volatility estimates for all metrics")

        st.markdown("\n**DJ30 Stock Price Data:**")
        for category, metrics in dj30_descriptions.items():
            st.markdown(f"\n*{category}:*")
            for metric in metrics:
                st.markdown(f"  {metric}")

    st.markdown("---")

    # Date Range Selection
    with st.expander("📅 Analysis Date Range", expanded=False):
        st.markdown("**Select the time period for your analysis:**")
        st.caption("Choose start and end dates to focus your analysis on a specific period.")

        from datetime import date

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.transduction_start_date,
                min_value=date(2018, 1, 1),
                max_value=date(2025, 12, 31),
                key="date_start_picker"
            )
            st.session_state.transduction_start_date = start_date

        with col2:
            end_date = st.date_input(
                "End Date",
                value=st.session_state.transduction_end_date,
                min_value=date(2018, 1, 1),
                max_value=date(2025, 12, 31),
                key="date_end_picker"
            )
            st.session_state.transduction_end_date = end_date

        # Validate date range
        if start_date >= end_date:
            st.error("⚠️ Start date must be before end date!")
        else:
            st.success(f"✅ Analyzing data from {start_date} to {end_date}")

    # Transduction Column Selection
    with st.expander("🔧 Transduction Column Selection", expanded=False):
        st.markdown("**Select columns to include in transduction analysis:**")
        st.caption("Date column is always included. Select additional columns to analyze.")

        all_columns = get_merged_data_columns()
        if all_columns:
            # Search box to filter columns
            search_term = st.text_input("🔍 Search columns:", placeholder="Type to filter columns...", key="column_search")

            # Filter columns based on search
            if search_term:
                search_lower = search_term.lower()
                all_columns = [col for col in all_columns if search_lower in col.lower()]
                if not all_columns:
                    st.info("No columns match your search.")
                    st.stop()
            # Separate Date from other columns
            date_col = "Date"
            other_columns = [col for col in all_columns if col != date_col]

            # Group columns by category for better UX
            macro_cols = [col for col in other_columns if col in ['FEDFUNDS', 'TB3MS', 'T10Y3M', 'CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE', 'UNRATE', 'PAYEMS', 'INDPRO', 'RSAFS']]
            market_cols = [col for col in other_columns if col.startswith('^') or col in ['BTC-USD', 'GSG', 'DGS2', 'DGS10', 'DTWEXBGS', 'DCOILBRENTEU', 'GLD', 'US10Y2Y', 'Headlines']]
            dj30_price_cols = [col for col in other_columns if any(col.startswith(prefix) for prefix in ['open_', 'high_', 'low_', 'close_', 'adj_close_', 'volume_', 'dividend_'])]
            fundamental_cols = [col for col in other_columns if any(col.endswith(suffix) for suffix in ['_MEDEST', '_MEANEST', '_ACTUAL'])]
            news_cols = [col for col in other_columns if col.startswith('news_') or col.startswith('Earningcall_')]
            other_cols = [col for col in other_columns if col not in macro_cols + market_cols + dj30_price_cols + fundamental_cols + news_cols]

            # Initialize selected columns if empty
            if "selected_transduction_columns" not in st.session_state:
                st.session_state.selected_transduction_columns = []

            # Build selected list from session state first (to ensure it reflects programmatic changes)
            # This ensures programmatically set columns are included
            # Make a copy to avoid reference issues
            selected = list(st.session_state.selected_transduction_columns) if st.session_state.selected_transduction_columns else []

            # Category selection with checkboxes
            # Macro columns
            if macro_cols:
                with st.expander(f"📊 Macroeconomic Indicators ({len(macro_cols)})", expanded=False):
                    for col in sorted(macro_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # Market columns
            if market_cols:
                with st.expander(f"📈 Market Factors ({len(market_cols)})", expanded=False):
                    for col in sorted(market_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # DJ30 Price columns
            if dj30_price_cols:
                with st.expander(f"💹 DJ30 Stock Prices ({len(dj30_price_cols)})", expanded=False):
                    # Group by ticker for better organization
                    ticker_groups = {}
                    for col in dj30_price_cols:
                        parts = col.split('_')
                        if len(parts) >= 2:
                            ticker = parts[-1]  # Last part is usually ticker
                            if ticker not in ticker_groups:
                                ticker_groups[ticker] = []
                            ticker_groups[ticker].append(col)

                    for ticker in sorted(ticker_groups.keys()):
                        with st.expander(f"  {ticker} ({len(ticker_groups[ticker])} columns)", expanded=False):
                            for col in sorted(ticker_groups[ticker]):
                                # Use session state key for checkbox to maintain state
                                checkbox_key = f"col_{col}"
                                # Initialize from session state if not set
                                if checkbox_key not in st.session_state:
                                    st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                                is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                                # Update selected list based on checkbox state
                                if is_checked:
                                    if col not in selected:
                                        selected.append(col)
                                else:
                                    if col in selected:
                                        selected.remove(col)

            # Fundamental columns
            if fundamental_cols:
                with st.expander(f"🏢 Company Fundamentals ({len(fundamental_cols)})", expanded=False):
                    # Group by metric type
                    metric_groups = {}
                    for col in fundamental_cols:
                        parts = col.split('_')
                        if len(parts) >= 2:
                            metric = parts[0]  # First part is metric
                            if metric not in metric_groups:
                                metric_groups[metric] = []
                            metric_groups[metric].append(col)

                    for metric in sorted(metric_groups.keys()):
                        with st.expander(f"  {metric} ({len(metric_groups[metric])} columns)", expanded=False):
                            for col in sorted(metric_groups[metric]):
                                # Use session state key for checkbox to maintain state
                                checkbox_key = f"col_{col}"
                                # Initialize from session state if not set
                                if checkbox_key not in st.session_state:
                                    st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                                is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                                # Update selected list based on checkbox state
                                if is_checked:
                                    if col not in selected:
                                        selected.append(col)
                                else:
                                    if col in selected:
                                        selected.remove(col)

            # News columns
            if news_cols:
                with st.expander(f"📰 News & Earnings Calls ({len(news_cols)})", expanded=False):
                    for col in sorted(news_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # Other columns
            if other_cols:
                with st.expander(f"🔹 Other Columns ({len(other_cols)})", expanded=False):
                    for col in sorted(other_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # Always update session state with current selection (removes duplicates)
            # This ensures the session state reflects both programmatic changes and user interactions
            final_selected = list(set(selected))
            st.session_state.selected_transduction_columns = final_selected


            # Show summary - use the final selected count
            # Use session state for display to ensure consistency
            total_selected = len(st.session_state.selected_transduction_columns)
            st.caption(f"📊 **{total_selected} columns selected** (Date always included)")

            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All", use_container_width=True):
                    st.session_state.selected_transduction_columns = other_columns
                    st.rerun()
            with col2:
                if st.button("❌ Clear All", use_container_width=True):
                    st.session_state.selected_transduction_columns = []
                    st.rerun()
        else:
            st.warning("Could not load column names from merged_data.csv")

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent_logs = ""
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    st.caption("Click any question to load it and auto-select relevant columns")

    # 5 carefully chosen diagnostic questions with manually pre-selected columns
    questions = [
        "Explain how AMZN and AAPL's strategy shifted over time, and any major investments they made from 2020 onwards",
        "Why did the stock price of AAPL drop in March 2020?",
        "What were the key factors behind NVDA's stock price surge in 2023?",
        "Analyze the relationship between MSFT's earnings announcements and its stock price movements from 2020-2023",
        "What market events and company-specific news drove JPM's volatility during the 2022-2023 period?"
    ]

    for i, q in enumerate(questions, 1):
        if st.button(f"{i}. {q}", key=f"q_{i}", use_container_width=True):
            set_example_question(q)

# Page selector in sidebar
with st.sidebar:
    st.markdown("### 📑 Navigation")
    page = st.radio(
        "Select Page",
        ["Chat", "Transduction Flow"],
        index=0 if st.session_state.current_page == "Chat" else 1,
        key="page_selector"
    )
    st.session_state.current_page = page
    st.markdown("---")

# Agent Logs Sidebar (Left) - Always visible
with st.sidebar:
    st.markdown("#### 🔍 Agent Thought Process")
    st.caption("View the agent's reasoning and tool calls in real-time")

    if st.session_state.agent_logs:
        # Show logs in expandable section
        with st.expander("📜 View Agent Logs", expanded=st.session_state.show_logs):
            st.code(st.session_state.agent_logs, language="text")

    else:
        st.info("💡 Agent logs will appear here once you ask a question. You'll be able to see the agent's tool usage, reasoning process, and decision-making in real-time!")

# Main content - Show different pages based on selection
if st.session_state.current_page == "Transduction Flow":
    display_transduction_flow()
else:
    # Chat page
    st.markdown('<div class="main-header">Transduction Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about macroeconomic data, market factors, and company fundamentals from 2018 to present</div>', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.container():
                st.markdown('<div class="chat-message user-message">', unsafe_allow_html=True)
                st.markdown("**You:**")
                st.markdown(content)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                st.markdown("**🤖 Analyst:**")
                # Escape special markdown characters to prevent rendering issues
                escaped_content = escape_markdown_special_chars(content)
                st.markdown(escaped_content)
                st.markdown('</div>', unsafe_allow_html=True)

    # User input
    with st.container():
        # If clear_input flag is set, reset the text area
        if st.session_state.clear_input:
            st.session_state.user_input = ""
            st.session_state.clear_input = False

        user_input = st.text_area(
            "Ask your question:",
            placeholder="E.g., How did market volatility change during the 2020 pandemic?",
            height=100,
            key="user_input"
        )

        col1, col2 = st.columns([8, 1])
        with col2:
            submit_button = st.button("Analyze", type="primary")

    # Process user input
    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Show loading state
        with st.spinner("Thinking..."):
            try:
                # Run analysis with conversation history for context and capture logs
                # Pass selected columns and date range for transduction filtering
                selected_cols = st.session_state.get("selected_transduction_columns", [])
                start_date = st.session_state.get("transduction_start_date")
                end_date = st.session_state.get("transduction_end_date")

                response = run_analysis_with_logs(
                    user_input,
                    selected_columns=selected_cols if selected_cols else None,
                    start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                    end_date=end_date.strftime("%Y-%m-%d") if end_date else None
                )

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

                # Clear previous transduction flow when new analysis starts
                # The new flow will be set by the transduction tool
                # This ensures we only show the latest flow

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Please try rephrasing your question."
                })

        # Set flag to clear input on next run
        st.session_state.clear_input = True

        # Rerun to display new messages
        st.rerun()

