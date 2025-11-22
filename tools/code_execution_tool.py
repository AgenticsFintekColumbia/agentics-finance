"""
Custom Code Execution Tool for Deep Research

This tool allows the agent to execute Python code with better error handling
and output formatting than the default CodeInterpreterTool.
"""

from crewai.tools import tool
import sys
from io import StringIO
import traceback
import os


def _execute_python_code_impl(code: str) -> str:
    """
    Internal implementation of Python code execution.

    This runs Python code in a controlled environment and captures:
    - Standard output (print statements)
    - Standard error (error messages)
    - Any exceptions raised

    Args:
        code: Python code to execute as a string

    Returns:
        String containing the execution output or error message
    """
    # Safety Check: Static Analysis using AST
    try:
        import ast
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Check for forbidden imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    if name.name.split('.')[0] in ['os', 'sys', 'subprocess', 'shutil']:
                        return f"❌ Safety Error: Import of '{name.name}' is forbidden."

            # Check for forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'exec', 'eval', '__import__', 'input']:
                        return f"❌ Safety Error: Call to '{node.func.id}()' is forbidden."
                elif isinstance(node.func, ast.Attribute):
                    # Block subprocess.run, os.system etc if they slipped through import checks
                    # (Though import check should catch them, this is double safety)
                    if node.func.attr in ['system', 'popen', 'run', 'spawn']:
                        return f"❌ Safety Error: Call to method '{node.func.attr}' is forbidden."

    except SyntaxError as e:
        return f"❌ Syntax Error in code: {e}"
    except Exception as e:
        return f"❌ Safety Check Failed: {e}"

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = StringIO()
    redirected_error = StringIO()

    try:
        # Redirect output
        sys.stdout = redirected_output
        sys.stderr = redirected_error

        # Create a safe execution environment with common libraries
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': None,
            'np': None,
            'plt': None,
            'go': None,
        }

        # Try to import common libraries
        try:
            import pandas as pd
            exec_globals['pd'] = pd
        except ImportError:
            pass

        try:
            import numpy as np
            exec_globals['np'] = np
        except ImportError:
            pass

        try:
            import matplotlib.pyplot as plt
            exec_globals['plt'] = plt
        except ImportError:
            pass

        try:
            import plotly.graph_objects as go
            exec_globals['go'] = go
        except ImportError:
            pass

        # Add helper function to save figures from within code
        def save_figure(fig, filename=None):
            """
            Helper function to save Plotly figures from within execute_python_code().

            Args:
                fig: Plotly Figure object
                filename: Optional filename

            Returns:
                String with Visualization ID
            """
            return _save_plotly_figure_impl(fig.to_json(), filename)

        # Add helper function to get news sentiment
        def get_news_sentiment(start_date=None, end_date=None):
            """
            Helper function to get daily news sentiment scores.
            Returns a DataFrame with 'Date' and 'sentiment_score' (-1 to 1).
            """
            try:
                from textblob import TextBlob
                import ast

                # Load market data which contains headlines
                df = pd.read_csv('./data/market_factors_new.csv')
                df['Date'] = pd.to_datetime(df['Date'])

                # Filter by date if provided
                if start_date:
                    df = df[df['Date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['Date'] <= pd.to_datetime(end_date)]

                sentiment_scores = []
                dates = []

                for _, row in df.iterrows():
                    headlines_raw = row.get('Headlines')
                    if pd.isna(headlines_raw):
                        continue

                    try:
                        # Parse stringified list
                        headlines = ast.literal_eval(str(headlines_raw))
                        if not isinstance(headlines, list):
                            headlines = [str(headlines_raw)]

                        # Calculate average sentiment for the day
                        day_scores = []
                        for h in headlines:
                            if h and isinstance(h, str):
                                blob = TextBlob(h)
                                day_scores.append(blob.sentiment.polarity)

                        if day_scores:
                            avg_score = sum(day_scores) / len(day_scores)
                            sentiment_scores.append(avg_score)
                            dates.append(row['Date'])
                    except:
                        continue

                result_df = pd.DataFrame({
                    'Date': dates,
                    'sentiment_score': sentiment_scores
                })
                result_df.set_index('Date', inplace=True)
                return result_df
            except Exception as e:
                print(f"Error calculating sentiment: {e}")
                return pd.DataFrame()

        exec_globals['save_figure'] = save_figure
        exec_globals['get_news_sentiment'] = get_news_sentiment

        # Execute the code
        exec(code, exec_globals)

        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get the output
        output = redirected_output.getvalue()
        errors = redirected_error.getvalue()

        # Truncate very long outputs to prevent context overflow
        # Reduced to 1800 for better Gemini compatibility in multi-step queries
        MAX_OUTPUT_LENGTH = 1800  # characters
        output_truncated = False
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH]
            output_truncated = True

        if errors:
            result = f"⚠️ Code executed with warnings:\n\n{errors}\n\nOutput:\n{output if output else '(no output)'}"
        elif output:
            result = f"✅ Code executed successfully:\n\n{output}"
        else:
            result = "✅ Code executed successfully (no output generated)"

        # Add truncation notice
        if output_truncated:
            result += f"\n\n⚠️ Output truncated (exceeded {MAX_OUTPUT_LENGTH} characters). Consider printing only summaries."

        return result

    except Exception as e:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get any output that was generated before the error
        partial_output = redirected_output.getvalue()
        error_msg = redirected_error.getvalue()

        # Format the error message (truncated for context management)
        error_trace = traceback.format_exc()
        error_type = type(e).__name__
        error_str = str(e)

        # Truncate error message if it's too long
        # Reduced for better context management in multi-step queries
        MAX_ERROR_LENGTH = 1000  # Even shorter for errors
        if len(error_str) > MAX_ERROR_LENGTH:
            error_str = error_str[:MAX_ERROR_LENGTH] + "... (truncated)"

        # Get only the last few lines of traceback (most relevant)
        trace_lines = error_trace.split('\n')
        if len(trace_lines) > 15:
            # Keep first 2 lines (File and line number) and last 10 lines (actual error)
            trace_lines = trace_lines[:2] + ['  ... (middle of traceback truncated) ...'] + trace_lines[-10:]
            error_trace = '\n'.join(trace_lines)

        # Provide specific, actionable fix suggestions based on error type
        fix_suggestion = _get_fix_suggestion(error_type, error_str)

        result = f"❌ Error during code execution:\n\n"
        result += f"Error Type: {error_type}\n"
        result += f"Error Message: {error_str}\n\n"

        if partial_output:
            # Truncate partial output too
            if len(partial_output) > 500:
                partial_output = partial_output[:500] + "... (truncated)"
            result += f"Partial Output (before error):\n{partial_output}\n\n"

        result += f"Traceback (simplified):\n{error_trace}\n\n"
        result += f"🔧 HOW TO FIX:\n{fix_suggestion}\n\n"
        result += "You can retry by writing corrected code in your next tool call."

        return result


def _get_fix_suggestion(error_type: str, error_message: str) -> str:
    """
    Provide specific fix suggestions based on error type.
    Helps the agent understand how to correct the code and retry.
    """
    error_message_lower = error_message.lower()

    # Type conversion errors
    if error_type == "TypeError" and "could not convert" in error_message_lower:
        if "numeric" in error_message_lower or "float" in error_message_lower:
            return (
                "Data type mismatch - trying to perform numeric operations on non-numeric data.\n"
                "Fix: Convert to numeric first:\n"
                "  df['column'] = pd.to_numeric(df['column'], errors='coerce')  # Converts, sets invalid to NaN\n"
                "  df = df.dropna(subset=['column'])  # Remove NaN values\n"
                "Or check data types: df.dtypes"
            )
        elif "datetime" in error_message_lower or "date" in error_message_lower:
            return (
                "Date parsing error.\n"
                "Fix: Convert to datetime first:\n"
                "  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')\n"
                "  df = df.dropna(subset=['Date'])"
            )

    # Key errors (column not found)
    elif error_type == "KeyError":
        return (
            "Column not found in dataframe.\n"
            "Fix:\n"
            "1. Check available columns: print(df.columns.tolist())\n"
            "2. Use exact column name (case-sensitive)\n"
            "3. Check for typos or extra spaces\n"
            "4. Verify you're using the correct dataset"
        )

    # File not found
    elif error_type == "FileNotFoundError":
        return (
            "File not found.\n"
            "Fix:\n"
            "1. Use correct path: './data/filename.csv'\n"
            "2. Check available files in dataset context\n"
            "3. Verify filename spelling"
        )

    # Index errors
    elif error_type == "IndexError":
        return (
            "Index out of range.\n"
            "Fix:\n"
            "1. Check dataframe length: print(len(df))\n"
            "2. Use .iloc[] for position-based indexing\n"
            "3. Ensure data exists before accessing"
        )

    # Value errors
    elif error_type == "ValueError":
        if "empty" in error_message_lower:
            return (
                "Operation on empty data.\n"
                "Fix:\n"
                "1. Check if dataframe is empty: if len(df) > 0:\n"
                "2. Verify filtering didn't remove all data\n"
                "3. Add data validation before operations"
            )
        elif "shape" in error_message_lower or "dimension" in error_message_lower:
            return (
                "Shape mismatch error.\n"
                "Fix:\n"
                "1. Check array shapes: print(arr.shape)\n"
                "2. Ensure dimensions align for operations\n"
                "3. Use reshape if needed: arr.reshape()"
            )

    # Attribute errors
    elif error_type == "AttributeError":
        return (
            "Attribute or method not found.\n"
            "Fix:\n"
            "1. Check object type: print(type(obj))\n"
            "2. Verify you're calling the right method\n"
            "3. Check pandas version compatibility"
        )

    # Zero division
    elif error_type == "ZeroDivisionError":
        return (
            "Division by zero.\n"
            "Fix:\n"
            "1. Check for zeros: df[df['column'] == 0]\n"
            "2. Add condition: if value != 0:\n"
            "3. Use np.where or handle explicitly"
        )

    # Memory errors
    elif error_type == "MemoryError":
        return (
            "Out of memory.\n"
            "Fix:\n"
            "1. Filter data to smaller date range\n"
            "2. Select only needed columns: df[['col1', 'col2']]\n"
            "3. Use sampling: df.sample(n=10000)\n"
            "4. Process in chunks"
        )

    # General fallback
    else:
        return (
            "General error occurred.\n"
            "Fix:\n"
            "1. Check data types: df.dtypes\n"
            "2. Handle missing data: df.dropna() or df.fillna()\n"
            "3. Add try-except for debugging:\n"
            "   try:\n"
            "       # your code\n"
            "   except Exception as e:\n"
            "       print(f'Debug: {e}')\n"
            "4. Print intermediate results to identify issue"
        )


def _save_plotly_figure_impl(figure_json: str, filename: str = None) -> str:
    """
    Internal implementation of Plotly figure saving.

    Args:
        figure_json: The Plotly figure as a JSON string (use fig.to_json())
        filename: Optional name for the file (auto-generated if not provided)

    Returns:
        Message with Visualization ID and file path
    """
    try:
        import plotly
        import json
        from datetime import datetime
        import random
        import string

        # Ensure visualizations directory exists
        viz_dir = './visualizations'
        os.makedirs(viz_dir, exist_ok=True)

        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        viz_id = f"viz_{timestamp}_{random_id}"
        filepath = os.path.join(viz_dir, f"{viz_id}.json")

        # Parse and save the figure
        fig_dict = json.loads(figure_json)

        # Wrap in a container for Streamlit to recognize it
        viz_data = {
            "type": "plotly_custom",  # Custom type for Deep Research visualizations
            "plotly_json": fig_dict,  # Raw Plotly JSON
            "filename": filename or "custom_visualization"
        }

        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(viz_data, f)

        return f"✅ Visualization saved successfully!\nVisualization ID: {viz_id}\nFile path: {filepath}"

    except Exception as e:
        return f"❌ Error saving visualization: {type(e).__name__}: {str(e)}"


# Create the CrewAI tool wrapper for code execution
@tool("Execute Python Code")
def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the output.

    This tool runs Python code in a controlled environment and captures output.
    Has pandas, numpy, matplotlib, plotly pre-imported as pd, np, plt, go.
    All datasets are in './data/' directory.

    Args:
        code: Python code to execute as a string

    Example:
        code = '''
import pandas as pd
df = pd.read_csv('./data/dj30_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
aapl_2023 = df[(df['Ticker'] == 'AAPL') & (df['Date'].dt.year == 2023)]
avg_return = aapl_2023['Daily_Return'].mean()
print(f"Average AAPL return in 2023: {avg_return:.4%}")
        '''
    """
    return _execute_python_code_impl(code)


# Create the CrewAI tool wrapper for saving visualizations
@tool("Save Plotly Visualization")
def save_plotly_figure(figure_json: str, filename: str = None) -> str:
    """
    Save a Plotly figure to the visualizations directory.

    Args:
        figure_json: The Plotly figure as a JSON string (use fig.to_json())
        filename: Optional name (auto-generated if not provided)

    Returns:
        Visualization ID that can be displayed in Streamlit

    Example:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[4,5,6])])
        save_plotly_figure(fig.to_json())
    """
    return _save_plotly_figure_impl(figure_json, filename)

