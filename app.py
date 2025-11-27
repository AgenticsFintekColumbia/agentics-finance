"""
Streamlit application for financial data Q&A with CrewAI agents.
"""

import streamlit as st
import os
import json
import glob
import re
import sys
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from agents import run_analysis, get_tool_categories
from utils import get_data_summary, get_column_descriptions, get_firm_data_summary, get_firm_column_descriptions, get_dj30_data_summary, get_dj30_column_descriptions

# Page configuration
st.set_page_config(
    page_title="Financial Data Analyst",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_visualizations" not in st.session_state:
    st.session_state.current_visualizations = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = ""

if "current_plan" not in st.session_state:
    st.session_state.current_plan = ""

if "show_logs" not in st.session_state:
    st.session_state.show_logs = True

if "enabled_tool_categories" not in st.session_state:
    # Enable all categories by default
    st.session_state.enabled_tool_categories = list(get_tool_categories().keys())

if "deep_research_mode" not in st.session_state:
    st.session_state.deep_research_mode = True

# Configuration: Maximum number of messages to keep in history
# Prevents context overflow and memory issues in long conversations
# Each exchange = 1 user message + 1 assistant message = 2 messages
MAX_MESSAGES = 10  # Keep last 10 exchanges (increase if needed, but may cause agent failures)


def truncate_conversation_history():
    """
    Truncate conversation history to prevent context overflow and memory issues.
    Keeps only the most recent messages to maintain performance.
    """
    if len(st.session_state.messages) > MAX_MESSAGES:
        # Keep only the most recent messages
        messages_to_remove = st.session_state.messages[:-MAX_MESSAGES]
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

        # Clean up visualizations from removed messages
        removed_viz_ids = set()
        for msg in messages_to_remove:
            if "visualizations" in msg and msg["visualizations"]:
                removed_viz_ids.update(msg["visualizations"])

        # Delete visualization files for removed messages
        if removed_viz_ids:
            viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
            for viz_id in removed_viz_ids:
                viz_file = os.path.join(viz_dir, f"{viz_id}.json")
                try:
                    if os.path.exists(viz_file):
                        os.remove(viz_file)
                except:
                    pass


def clean_old_visualizations():
    """Remove visualization files that are no longer referenced in conversation history."""
    viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    if os.path.exists(viz_dir):
        # Collect all visualization IDs from entire conversation history
        all_referenced_viz_ids = set()

        for message in st.session_state.messages:
            if "visualizations" in message and message["visualizations"]:
                all_referenced_viz_ids.update(message["visualizations"])

        # Also include current visualizations
        all_referenced_viz_ids.update(st.session_state.current_visualizations)

        # Get all visualization files
        all_viz_files = glob.glob(os.path.join(viz_dir, "*.json"))

        # Build set of files to keep
        files_to_keep = {
            os.path.join(viz_dir, f"{viz_id}.json")
            for viz_id in all_referenced_viz_ids
        }

        # Delete only files not referenced in conversation history
        for viz_file in all_viz_files:
            if viz_file not in files_to_keep:
                try:
                    os.remove(viz_file)
                except:
                    pass


def load_visualization(viz_id: str):
    """Load and render a visualization from JSON file."""
    viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    viz_file = os.path.join(viz_dir, f"{viz_id}.json")

    if not os.path.exists(viz_file):
        st.warning(f"Visualization {viz_id} not found.")
        return

    try:
        with open(viz_file, 'r') as f:
            viz_config = json.load(f)

        viz_type = viz_config.get("type")

        # Pass viz_id to rendering functions for unique keys
        if viz_type == "time_series":
            render_time_series(viz_config, viz_id)
        elif viz_type == "correlation_heatmap":
            render_correlation_heatmap(viz_config, viz_id)
        elif viz_type == "volatility_plot":
            render_volatility_plot(viz_config, viz_id)
        elif viz_type == "distribution":
            render_distribution(viz_config, viz_id)
        elif viz_type == "scatter":
            render_scatter(viz_config, viz_id)
        elif viz_type == "comparative_performance":
            render_comparative_performance(viz_config, viz_id)
        elif viz_type == "moving_average":
            render_moving_average(viz_config, viz_id)
        elif viz_type == "drawdown":
            render_drawdown(viz_config, viz_id)
        elif viz_type == "multi_indicator":
            render_multi_indicator(viz_config, viz_id)
        elif viz_type == "company_comparison":
            render_company_comparison(viz_config, viz_id)
        elif viz_type == "fundamental_time_series":
            render_fundamental_time_series(viz_config, viz_id)
        elif viz_type == "valuation_scatter":
            render_valuation_scatter(viz_config, viz_id)
        elif viz_type == "portfolio_recommendation":
            render_portfolio_recommendation(viz_config, viz_id)
        elif viz_type == "price_chart":
            render_price_chart(viz_config, viz_id)
        elif viz_type == "performance_comparison":
            render_performance_comparison(viz_config, viz_id)
        elif viz_type == "volatility_chart":
            render_volatility_chart(viz_config, viz_id)
        elif viz_type == "volatility_portfolio":
            render_volatility_portfolio(viz_config, viz_id)
        elif viz_type == "momentum_portfolio":
            render_momentum_portfolio(viz_config, viz_id)
        elif viz_type == "sector_portfolio":
            render_sector_portfolio(viz_config, viz_id)
        elif viz_type == "gmv_efficient_frontier":
            render_gmv_efficient_frontier(viz_config, viz_id)
        elif viz_type == "plotly_custom":
            # Handle custom Plotly visualizations from Deep Research mode
            render_custom_plotly(viz_config, viz_id)
        else:
            st.warning(f"Unknown visualization type: {viz_type}")
    except Exception as e:
        st.error(f"Error rendering visualization {viz_id}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_custom_plotly(config: dict, viz_id: str = None):
    """Render custom Plotly visualization from Deep Research mode."""
    try:
        import plotly.graph_objects as go

        # Extract the raw Plotly JSON
        plotly_json = config.get("plotly_json")

        if not plotly_json:
            st.error("No Plotly data found in visualization config")
            return

        # Create figure from JSON
        fig = go.Figure(plotly_json)

        # Use unique key for this visualization
        unique_key = f"{viz_id}_custom_plotly" if viz_id else "custom_plotly"

        # Display the figure
        st.plotly_chart(fig, use_container_width=True, key=unique_key)

    except Exception as e:
        st.error(f"Error rendering custom Plotly visualization: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_time_series(config: dict, viz_id: str = None):
    """Render time series plot with support for dual y-axes when scales differ significantly."""
    df = pd.DataFrame(config["data"])
    indicators = config.get("indicators", df["indicator"].unique().tolist())

    # Indicators that should be on secondary axis (volatility indices, percentages, etc.)
    secondary_axis_indicators = ['^VIX', 'UNRATE', 'FEDFUNDS', 'DGS10', 'DGS2', 'CPIAUCSL']

    # Determine if we need dual axes
    # Check if we have both a secondary axis indicator and a primary axis indicator
    has_secondary = any(ind in secondary_axis_indicators for ind in indicators)
    has_primary = any(ind not in secondary_axis_indicators for ind in indicators)
    use_dual_axes = has_secondary and has_primary and len(indicators) > 1

    if use_dual_axes:
        # Use make_subplots with secondary_y for better visualization of different scales
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, indicator in enumerate(indicators):
            indicator_df = df[df["indicator"] == indicator]
            use_secondary = indicator in secondary_axis_indicators

            fig.add_trace(
                go.Scatter(
                    x=indicator_df["date"],
                    y=indicator_df["value"],
                    name=indicator,
                    line=dict(color=colors[i % len(colors)]),
                    mode='lines'
                ),
                secondary_y=use_secondary
            )

        # Update axes labels
        primary_indicators = [ind for ind in indicators if ind not in secondary_axis_indicators]
        secondary_indicators = [ind for ind in indicators if ind in secondary_axis_indicators]

        fig.update_yaxes(
            title_text=", ".join(primary_indicators) if primary_indicators else "Value",
            secondary_y=False
        )
        fig.update_yaxes(
            title_text=", ".join(secondary_indicators) if secondary_indicators else "Value",
            secondary_y=True
        )

        fig.update_xaxes(title_text="Date")

        fig.update_layout(
            title=config["title"],
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
    else:
        # Use standard plotly express for single scale
        fig = px.line(
            df,
            x="date",
            y="value",
            color="indicator",
            title=config["title"],
            labels={"date": "Date", "value": "Value", "indicator": "Indicator"}
        )

        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_time_series" if viz_id else None)


def render_correlation_heatmap(config: dict, viz_id: str = None):
    """Render correlation heatmap."""
    df = pd.DataFrame(config["data"])

    # Pivot data for heatmap
    pivot_df = df.pivot(index="y", columns="x", values="correlation")

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdBu",
        zmid=0,
        text=pivot_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=config["title"],
        xaxis_title="",
        yaxis_title="",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_heatmap" if viz_id else None)


def render_volatility_plot(config: dict, viz_id: str = None):
    """Render volatility plot with dual y-axes."""
    df = pd.DataFrame(config["data"])

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add value trace
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            name=config["indicator"],
            line=dict(color="#1f77b4")
        ),
        secondary_y=False,
    )

    # Add volatility trace if available
    if "volatility" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["volatility"],
                name=f"Volatility (rolling {config['window']}d)",
                line=dict(color="#ff7f0e", dash="dash")
            ),
            secondary_y=True,
        )

    # Update layout
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=config["indicator"], secondary_y=False)
    fig.update_yaxes(title_text="Volatility", secondary_y=True)

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_volatility" if viz_id else None)


def render_distribution(config: dict, viz_id: str = None):
    """Render distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=config["data"],
        nbinsx=50,
        name=config["indicator"],
        marker_color="#1f77b4"
    ))

    # Add mean line
    mean = config["stats"]["mean"]
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean:.2f}"
    )

    fig.update_layout(
        title=config["title"],
        xaxis_title=config["indicator"],
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_distribution" if viz_id else None)


def render_scatter(config: dict, viz_id: str = None):
    """Render scatter plot."""
    df = pd.DataFrame(config["data"])

    fig = px.scatter(
        df,
        x="x",
        y="y",
        title=config["title"],
        labels={"x": config["x_indicator"], "y": config["y_indicator"]},
        hover_data=["date"]
    )

    # Add trendline
    fig.update_traces(marker=dict(size=5, opacity=0.6))

    # Add correlation text
    correlation = config.get("correlation", 0)
    fig.add_annotation(
        text=f"Correlation: {correlation:.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_scatter" if viz_id else None)


def render_comparative_performance(config: dict, viz_id: str = None):
    """Render comparative performance chart (normalized to 100)."""
    df = pd.DataFrame(config["data"])

    fig = px.line(
        df,
        x="date",
        y="value",
        color="indicator",
        title=config["title"],
        labels={"date": "Date", "value": "Normalized Value (Base=100)", "indicator": "Indicator"}
    )

    # Add horizontal line at 100
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Start")

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_comparative" if viz_id else None)


def render_moving_average(config: dict, viz_id: str = None):
    """Render moving average chart."""
    df = pd.DataFrame(config["data"])

    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["price"],
        name=config["indicator"],
        line=dict(color="#1f77b4", width=2)
    ))

    # Add moving averages
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, window in enumerate(config["windows"]):
        ma_col = f"MA_{window}"
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df[ma_col],
                name=f"MA {window}",
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dash")
            ))

    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_moving_avg" if viz_id else None)


def render_drawdown(config: dict, viz_id: str = None):
    """Render drawdown chart."""
    df = pd.DataFrame(config["data"])

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(config["indicator"], "Drawdown from Peak"),
        row_heights=[0.6, 0.4]
    )

    # Add price
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["price"],
            name=config["indicator"],
            line=dict(color="#1f77b4")
        ),
        row=1, col=1
    )

    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown_pct"],
            name="Drawdown %",
            fill='tozeroy',
            line=dict(color="#d62728")
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_drawdown" if viz_id else None)


def render_multi_indicator(config: dict, viz_id: str = None):
    """Render multi-indicator dashboard with subplots."""
    indicators = config["indicators"]
    data = config["data"]

    # Create subplots
    fig = make_subplots(
        rows=len(indicators),
        cols=1,
        shared_xaxes=True,
        subplot_titles=indicators,
        vertical_spacing=0.05
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Add each indicator
    for i, indicator in enumerate(indicators):
        indicator_data = data[indicator]
        df = pd.DataFrame(indicator_data)

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                name=indicator,
                line=dict(color=colors[i % len(colors)]),
                showlegend=False
            ),
            row=i+1, col=1
        )

        fig.update_yaxes(title_text=indicator, row=i+1, col=1)

    fig.update_xaxes(title_text="Date", row=len(indicators), col=1)

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        height=250 * len(indicators)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_multi_indicator" if viz_id else None)


def render_company_comparison(config: dict, viz_id: str = None):
    """Render company comparison bar chart."""
    data = config["data"]
    metrics = config["metrics"]

    # Create subplots for each metric
    num_metrics = len(metrics)
    fig = make_subplots(
        rows=num_metrics,
        cols=1,
        subplot_titles=metrics,
        vertical_spacing=0.08
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, metric in enumerate(metrics):
        tickers = [d["ticker"] for d in data]
        values = [d.get(metric) for d in data]

        # Filter out None values
        filtered_data = [(t, v) for t, v in zip(tickers, values) if v is not None]
        if not filtered_data:
            continue

        tickers_filtered, values_filtered = zip(*filtered_data)

        fig.add_trace(
            go.Bar(
                x=list(tickers_filtered),
                y=list(values_filtered),
                name=metric,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=i+1, col=1
        )

        fig.update_yaxes(title_text=metric, row=i+1, col=1)

    fig.update_xaxes(title_text="Company", row=num_metrics, col=1)

    fig.update_layout(
        title=config["title"],
        height=300 * num_metrics,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_company_comp" if viz_id else None)


def render_fundamental_time_series(config: dict, viz_id: str = None):
    """Render fundamental time series plot."""
    df = pd.DataFrame(config["data"])
    metrics = config["metrics"]

    # Check if we need dual axes based on scale differences
    if len(metrics) > 1:
        # Simple heuristic: if ranges differ by more than 10x, use dual axes
        ranges = {}
        for metric in metrics:
            metric_data = df[df["metric"] == metric]["value"]
            if not metric_data.empty:
                ranges[metric] = metric_data.max() - metric_data.min()

        if ranges:
            max_range = max(ranges.values())
            min_range = min(ranges.values())
            use_dual_axes = max_range / min_range > 10 if min_range > 0 else False
        else:
            use_dual_axes = False
    else:
        use_dual_axes = False

    if use_dual_axes and len(metrics) == 2:
        # Use dual y-axes for 2 metrics with different scales
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = ["#1f77b4", "#ff7f0e"]
        for i, metric in enumerate(metrics):
            metric_df = df[df["metric"] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_df["date"],
                    y=metric_df["value"],
                    name=metric,
                    line=dict(color=colors[i]),
                    mode='lines+markers'
                ),
                secondary_y=(i == 1)
            )

        fig.update_yaxes(title_text=metrics[0], secondary_y=False)
        fig.update_yaxes(title_text=metrics[1], secondary_y=True)
    else:
        # Standard plot
        fig = px.line(
            df,
            x="date",
            y="value",
            color="metric",
            title=config["title"],
            labels={"date": "Date", "value": "Value", "metric": "Metric"},
            markers=True
        )

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_fundamental_ts" if viz_id else None)


def render_valuation_scatter(config: dict, viz_id: str = None):
    """Render valuation scatter plot."""
    df = pd.DataFrame(config["data"])

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="ticker",
        title=config["title"],
        labels={"x": config["x_metric"], "y": config["y_metric"]},
        size_max=15
    )

    # Position labels above points
    fig.update_traces(
        textposition='top center',
        marker=dict(size=12, opacity=0.7)
    )

    # Add trendline
    if len(df) >= 2:
        z = np.polyfit(df["x"], df["y"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df["x"].min(), df["x"].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trendline',
                line=dict(dash='dash', color='gray')
            )
        )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_valuation" if viz_id else None)


def render_portfolio_recommendation(config: dict, viz_id: str = None):
    """Render portfolio recommendation chart."""
    long_positions = config["long_positions"]
    short_positions = config["short_positions"]

    # Display recommendation summary table
    st.markdown("### Portfolio Recommendations Summary")

    def color_rating(rating):
        """Return colored HTML for rating."""
        colors = {
            "Strong Buy": "#00C851",  # Green
            "Buy": "#4CAF50",         # Light Green
            "Hold": "#FFC107",        # Amber
            "Sell": "#FF5722",        # Deep Orange
            "Strong Sell": "#F44336"  # Red
        }
        color = colors.get(rating, "#757575")
        return f'<span style="color: {color}; font-weight: bold;">{rating}</span>'

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🟢 Long Positions**")
        for i, p in enumerate(long_positions):
            roe_val = f"{p.get('roe', 0):.1f}%" if p.get('roe') else "N/A"
            pe_val = f"{p.get('pe_ratio', 0):.1f}" if p.get('pe_ratio') else "N/A"
            rating_html = color_rating(p.get('rating', 'N/A'))

            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #4CAF50; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> - {rating_html}<br>
                <small>ROE: {roe_val} | P/E: {pe_val}</small>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🔴 Short Positions**")
        for i, p in enumerate(short_positions):
            roe_val = f"{p.get('roe', 0):.1f}%" if p.get('roe') else "N/A"
            pe_val = f"{p.get('pe_ratio', 0):.1f}" if p.get('pe_ratio') else "N/A"
            rating_html = color_rating(p.get('rating', 'N/A'))

            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> - {rating_html}<br>
                <small>ROE: {roe_val} | P/E: {pe_val}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create subplots for metrics
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Return on Equity (%)", "P/E Ratio", "EPS Growth (%)"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Combine data
    all_positions = []
    for pos in long_positions:
        all_positions.append({**pos, "position": "LONG", "color": "#2ca02c"})
    for pos in short_positions:
        all_positions.append({**pos, "position": "SHORT", "color": "#d62728"})

    if not all_positions:
        st.warning("No positions to display")
        return

    tickers = [p["ticker"] for p in all_positions]
    colors = [p["color"] for p in all_positions]

    # ROE
    roe_values = [p.get("roe") for p in all_positions]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=roe_values,
            marker_color=colors,
            name="ROE",
            showlegend=False
        ),
        row=1, col=1
    )

    # P/E Ratio
    pe_values = [p.get("pe_ratio") for p in all_positions]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=pe_values,
            marker_color=colors,
            name="P/E",
            showlegend=False
        ),
        row=2, col=1
    )

    # EPS Growth
    eps_growth_values = [p.get("eps_growth") for p in all_positions]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=eps_growth_values,
            marker_color=colors,
            name="EPS Growth",
            showlegend=False
        ),
        row=3, col=1
    )

    # Add legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#2ca02c'),
            name='LONG'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#d62728'),
            name='SHORT'
        )
    )

    fig.update_layout(
        title=config["title"],
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_portfolio" if viz_id else None)


def render_price_chart(config: dict, viz_id: str = None):
    """Render DJ30 price chart (candlestick, line, or OHLC)."""
    df = pd.DataFrame(config["data"])
    df['date'] = pd.to_datetime(df['date'])

    chart_type = config.get("chart_type", "candlestick")
    include_volume = config.get("include_volume", True)

    if include_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )
    else:
        fig = go.Figure()

    # Add price trace
    if chart_type == "candlestick":
        trace = go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        )
    elif chart_type == "ohlc":
        trace = go.Ohlc(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        )
    else:  # line
        trace = go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name="Price",
            line=dict(color="#1f77b4")
        )

    if include_volume:
        fig.add_trace(trace, row=1, col=1)
        # Add volume trace
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name="Volume", marker_color="#A9A9A9"),
            row=2, col=1
        )
    else:
        fig.add_trace(trace)

    fig.update_layout(
        title=config["title"],
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_price" if viz_id else None)


def render_performance_comparison(config: dict, viz_id: str = None):
    """Render DJ30 performance comparison chart."""
    series = config["series"]

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (ticker, data) in enumerate(series.items()):
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name=ticker,
            line=dict(color=colors[i % len(colors)])
        ))

    ylabel = "Normalized Price (Base=100)" if config.get("normalized", False) else "Price ($)"

    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_performance" if viz_id else None)


def render_volatility_chart(config: dict, viz_id: str = None):
    """Render DJ30 rolling volatility chart."""
    df = pd.DataFrame(config["data"])
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['volatility'],
        mode='lines',
        name=f"{config['window']}-day Rolling Volatility",
        line=dict(color="#ff7f0e"),
        fill='tozeroy'
    ))

    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_vol_chart" if viz_id else None)


def render_volatility_portfolio(config: dict, viz_id: str = None):
    """Render volatility-based portfolio recommendations."""
    long_positions = config.get("long_positions", [])
    short_positions = config.get("short_positions", [])
    portfolio_type = config.get("portfolio_type", "long_short")

    # Render title
    st.markdown(f"### {config.get('title', 'Volatility-Based Portfolio')}")

    # Determine layout based on what positions exist
    has_long = len(long_positions) > 0
    has_short = len(short_positions) > 0

    if has_long and has_short:
        # Two columns for long/short
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Long Positions**")
            label = "High Volatility" if portfolio_type == "long_short" or portfolio_type == "long_high_vol" else "Low Volatility"
            st.caption(label)
            for i, p in enumerate(long_positions):
                div_info = f" | Div: {p.get('dividend_yield', 0):.2f}%" if p.get('dividend_yield', 0) > 0 else ""
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #4CAF50; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%{div_info}</small>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**🔴 Short Positions**")
            st.caption("Low Volatility" if portfolio_type == "long_short" else "High Volatility")
            for i, p in enumerate(short_positions):
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

    elif has_long:
        # Only long positions - single column
        st.markdown("**🟢 Long Positions**")
        if portfolio_type == "long_low_vol":
            st.caption("Low Volatility - Defensive Strategy")
        else:
            st.caption("High Volatility - Aggressive Strategy")

        for i, p in enumerate(long_positions):
            div_info = f" | Div: {p.get('dividend_yield', 0):.2f}%" if p.get('dividend_yield', 0) > 0 else ""
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #4CAF50; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%{div_info}</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    elif has_short:
        # Only short positions - single column
        st.markdown("**🔴 Short Positions**")
        st.caption("High Volatility")

        for i, p in enumerate(short_positions):
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #F44336; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create volatility comparison bar chart
    tickers = [p['ticker'] for p in long_positions] + [p['ticker'] for p in short_positions]
    volatilities = [p['volatility'] for p in long_positions] + [p['volatility'] for p in short_positions]
    colors = ['#4CAF50'] * len(long_positions) + ['#F44336'] * len(short_positions)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=volatilities,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in volatilities],
        textposition='outside'
    ))

    fig.update_layout(
        title="Volatility Comparison",
        xaxis_title="Ticker",
        yaxis_title="Annualized Volatility (%)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_vol_portfolio" if viz_id else None)


def render_momentum_portfolio(config: dict, viz_id: str = None):
    """Render momentum-based portfolio recommendations."""
    long_positions = config.get("long_positions", [])
    short_positions = config.get("short_positions", [])
    portfolio_type = config.get("portfolio_type", "long_short")

    # Render title
    st.markdown(f"### {config.get('title', 'Momentum-Based Portfolio')}")

    # Determine layout based on what positions exist
    has_long = len(long_positions) > 0
    has_short = len(short_positions) > 0

    if has_long and has_short:
        # Two columns for long/short
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Long Positions (High Momentum)**")
            for i, p in enumerate(long_positions):
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #4CAF50; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**🔴 Short Positions (Low Momentum)**")
            for i, p in enumerate(short_positions):
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

    elif has_long:
        # Only long positions - single column
        st.markdown("**🟢 Long Positions**")
        st.caption("High Momentum - Trend Following Strategy")

        for i, p in enumerate(long_positions):
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #4CAF50; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    elif has_short:
        # Only short positions - single column
        st.markdown("**🔴 Short Positions**")
        st.caption("Low Momentum - Contrarian Strategy")

        for i, p in enumerate(short_positions):
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #F44336; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create momentum comparison bar chart
    tickers = [p['ticker'] for p in long_positions] + [p['ticker'] for p in short_positions]
    momentum = [p['momentum'] for p in long_positions] + [p['momentum'] for p in short_positions]
    colors = ['#4CAF50'] * len(long_positions) + ['#F44336'] * len(short_positions)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=momentum,
        marker_color=colors,
        text=[f"{m:+.1f}%" for m in momentum],
        textposition='outside'
    ))

    fig.update_layout(
        title="Momentum Comparison",
        xaxis_title="Ticker",
        yaxis_title="Cumulative Return (%)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_momentum" if viz_id else None)


def render_sector_portfolio(config: dict, viz_id: str = None):
    """Render sector-diversified portfolio."""
    positions = config["positions"]

    st.markdown("### Sector-Diversified Portfolio")

    # Group by sector
    sectors = {}
    for p in positions:
        sector = p.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(p)

    # Display by sector
    for sector, stocks in sectors.items():
        st.markdown(f"**{sector}**")
        for p in stocks:
            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #2196F3; background-color: var(--secondary-background-color);">
                <strong>{p['ticker']}</strong><br>
                <small>Return: {p.get('total_return', 0):+.2f}% | Sharpe: {p.get('sharpe_ratio', 0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create sector allocation pie chart
    sector_counts = {sector: len(stocks) for sector, stocks in sectors.items()}

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=list(sector_counts.keys()),
        values=list(sector_counts.values()),
        hole=0.3
    ))

    fig.update_layout(
        title="Sector Allocation",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_sector" if viz_id else None)


def render_gmv_efficient_frontier(config: dict, viz_id: str = None):
    """Render GMV portfolio efficient frontier visualization."""
    simulated_portfolios = config["simulated_portfolios"]
    gmv_portfolio = config["gmv_portfolio"]
    equal_weight_portfolio = config["equal_weight_portfolio"]
    metadata = config.get("metadata", {})

    st.markdown(f"### {config.get('title', 'GMV Portfolio Efficient Frontier')}")

    # Display key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "GMV Return",
            f"{gmv_portfolio['return']*100:.2f}%",
            help="Annualized expected return of GMV portfolio"
        )
        st.metric(
            "GMV Volatility",
            f"{gmv_portfolio['volatility']*100:.2f}%",
            help="Annualized volatility (risk) of GMV portfolio"
        )
        st.metric(
            "GMV Sharpe Ratio",
            f"{gmv_portfolio['sharpe']:.4f}",
            help="Risk-adjusted return metric"
        )

    with col2:
        st.metric(
            "Equal-Weight Return",
            f"{equal_weight_portfolio['return']*100:.2f}%"
        )
        st.metric(
            "Equal-Weight Volatility",
            f"{equal_weight_portfolio['volatility']*100:.2f}%"
        )
        st.metric(
            "Equal-Weight Sharpe",
            f"{equal_weight_portfolio['sharpe']:.4f}"
        )

    with col3:
        variance_reduction = ((equal_weight_portfolio['volatility'] - gmv_portfolio['volatility']) /
                              equal_weight_portfolio['volatility']) * 100
        st.metric(
            "Variance Reduction",
            f"{variance_reduction:.2f}%",
            help="Volatility reduction vs. equal-weight portfolio"
        )
        st.metric(
            "# of Assets",
            metadata.get('n_assets', 'N/A')
        )
        st.metric(
            "# of Simulations",
            f"{metadata.get('n_simulations', 0):,}"
        )

    st.markdown("---")

    # Create efficient frontier scatter plot
    sim_returns = [p['return'] * 100 for p in simulated_portfolios]  # Convert to percentage
    sim_volatilities = [p['volatility'] * 100 for p in simulated_portfolios]
    sim_sharpes = [p['sharpe'] for p in simulated_portfolios]

    fig = go.Figure()

    # Simulated portfolios (colored by Sharpe ratio)
    fig.add_trace(go.Scatter(
        x=sim_volatilities,
        y=sim_returns,
        mode='markers',
        name='Simulated Portfolios',
        marker=dict(
            size=4,
            color=sim_sharpes,
            colorscale='RdYlGn',  # Red (low) to Yellow to Green (high)
            colorbar=dict(
                title="Sharpe<br>Ratio",
                x=1.15
            ),
            showscale=True,
            opacity=0.6
        ),
        text=[f"Return: {r:.2f}%<br>Volatility: {v:.2f}%<br>Sharpe: {s:.4f}"
              for r, v, s in zip(sim_returns, sim_volatilities, sim_sharpes)],
        hovertemplate='%{text}<extra></extra>'
    ))

    # GMV Portfolio (highlighted)
    fig.add_trace(go.Scatter(
        x=[gmv_portfolio['volatility'] * 100],
        y=[gmv_portfolio['return'] * 100],
        mode='markers+text',
        name='GMV Portfolio',
        marker=dict(
            size=20,
            color='blue',
            symbol='star',
            line=dict(color='darkblue', width=2)
        ),
        text=['GMV'],
        textposition='top center',
        textfont=dict(size=12, color='blue', family='Arial Black'),
        hovertemplate=(
            f"<b>GMV Portfolio (Nodewise Lasso)</b><br>"
            f"Return: {gmv_portfolio['return']*100:.2f}%<br>"
            f"Volatility: {gmv_portfolio['volatility']*100:.2f}%<br>"
            f"Sharpe: {gmv_portfolio['sharpe']:.4f}<br>"
            "<extra></extra>"
        )
    ))

    # Equal-Weight Portfolio (for comparison)
    fig.add_trace(go.Scatter(
        x=[equal_weight_portfolio['volatility'] * 100],
        y=[equal_weight_portfolio['return'] * 100],
        mode='markers+text',
        name='Equal-Weight',
        marker=dict(
            size=15,
            color='orange',
            symbol='diamond',
            line=dict(color='darkorange', width=2)
        ),
        text=['Equal'],
        textposition='top center',
        textfont=dict(size=10, color='orange'),
        hovertemplate=(
            f"<b>Equal-Weight Portfolio</b><br>"
            f"Return: {equal_weight_portfolio['return']*100:.2f}%<br>"
            f"Volatility: {equal_weight_portfolio['volatility']*100:.2f}%<br>"
            f"Sharpe: {equal_weight_portfolio['sharpe']:.4f}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"Efficient Frontier: {metadata.get('n_simulations', 0):,} Simulated Portfolios",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='rgba(50,50,50,0.9)')
        ),
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.4)",
            bordercolor="rgba(200,200,200,0.3)",
            borderwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color='rgba(50,50,50,0.9)'
        )
    )

    # Add subtle grid
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(150,150,150,0.15)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(150,150,150,0.3)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(150,150,150,0.15)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(150,150,150,0.3)'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{viz_id}_efficient_frontier" if viz_id else None)

    # Display top holdings
    st.markdown("#### GMV Portfolio Top Holdings")
    weights = gmv_portfolio.get('weights', {})
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]

    col1, col2 = st.columns(2)
    for i, (asset, weight) in enumerate(sorted_weights):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"**{asset}**: {weight*100:.2f}%")

    # Explanation
    st.markdown("---")
    st.info(
        "📊 **Interpretation**: The blue star shows the GMV (Global Minimum Variance) portfolio "
        "computed via nodewise Lasso regression. It achieves the lowest volatility among all possible portfolios. "
        "Each gray dot represents a randomly simulated portfolio, colored by its Sharpe ratio (green = better risk-adjusted returns). "
        "The orange diamond shows an equal-weight benchmark for comparison."
    )


def extract_visualization_ids(response: str, start_time: float = 0) -> list:
    """
    Extract visualization IDs from agent response and file system.

    Args:
        response: The text response from the agent
        start_time: Timestamp when the generation started (to find new files)

    Returns:
        List of visualization IDs sorted by creation time
    """
    import re
    import glob
    import os
    from datetime import datetime

    # 1. Find all visualization files created AFTER the start time
    viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    new_viz_files = []

    if os.path.exists(viz_dir):
        for viz_file in glob.glob(os.path.join(viz_dir, "viz_*.json")):
            try:
                mtime = os.path.getmtime(viz_file)
                if mtime >= start_time:
                    viz_id = os.path.basename(viz_file).replace('.json', '')
                    new_viz_files.append((mtime, viz_id))
            except OSError:
                continue

    # Sort by creation time to preserve logical order
    new_viz_files.sort(key=lambda x: x[0])
    file_viz_ids = [v[1] for v in new_viz_files]

    # 2. Extract IDs explicitly mentioned in the text (fallback/legacy)
    text_viz_ids = re.findall(r'viz_\d{8}_\d{6}_[a-f0-9]{8}', response)

    # 3. Combine: File-detected IDs first (chronological), then any text-only IDs
    # Use dict.fromkeys to deduplicate while preserving order
    all_ids = list(dict.fromkeys(file_viz_ids + text_viz_ids))

    return all_ids


def run_analysis_with_logs(user_input: str, conversation_history: list, enabled_tool_categories: list = None) -> str:
    """Run analysis and capture stdout/stderr to display in logs."""
    # Create a StringIO object to capture output
    captured_output = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect stdout and stderr to capture output
        sys.stdout = captured_output
        sys.stderr = captured_output

        # Run the analysis with filtered tools
        response = run_analysis(user_input, conversation_history, enabled_tool_categories=enabled_tool_categories)

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


def run_deep_research_with_logs(user_input: str, plan_placeholder=None) -> str:
    """Run deep research analysis and capture stdout/stderr to display in logs."""
    # Create a StringIO object to capture output
    captured_output = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect stdout and stderr to capture output
        sys.stdout = captured_output
        sys.stderr = captured_output

        from agents.deep_research_analyst import generate_research_plan, execute_research_plan

        # 1. Planning Phase
        plan = ""
        if plan_placeholder:
            with plan_placeholder.container():
                # Use status for the planning phase
                with st.status("🧠 Generating Research Plan...", expanded=True) as status:
                    plan = generate_research_plan(user_input)
                    status.markdown(plan)
                    status.update(label="🧠 Research Plan", state="complete", expanded=False)
        else:
            plan = generate_research_plan(user_input)

        # Store the plan in session state for display
        st.session_state.current_plan = plan

        # 2. Execution Phase
        with st.spinner("🔬 Conducting deep research..."):
            response = execute_research_plan(user_input, plan)

        print("RESPONSE RESPONSE: ", response)

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


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Analysis Mode")

    # Auto-Routing Toggle
    auto_routing = st.toggle(
        "🤖 Auto-Routing",
        value=st.session_state.get("auto_routing", True),
        key="auto_routing_toggle",
        help="Automatically route queries to the best agent based on complexity."
    )
    st.session_state.auto_routing = auto_routing

    # Deep Research toggle (Manual Override)
    deep_research_enabled = st.toggle(
        "🔬 Deep Research Mode",
        value=st.session_state.deep_research_mode,
        disabled=st.session_state.auto_routing,
        help="Enable comprehensive multi-step analysis with code execution. "
             "The agent will decompose complex problems, write Python code dynamically, "
             "and provide detailed insights. Best for complex questions like portfolio "
             "optimization, strategy backtesting, or multi-factor analysis."
    )

    # Update session state if changed
    if deep_research_enabled != st.session_state.deep_research_mode:
        st.session_state.deep_research_mode = deep_research_enabled

    # Show mode description
    if st.session_state.deep_research_mode:
        st.info(
            "🔬 **Deep Research Active**\n\n"
            "The agent will:\n"
            "- Decompose problems into steps\n"
            "- Write & execute Python code\n"
            "- Generate custom visualizations\n"
            "- Provide comprehensive analysis\n\n"
            "⏱️ *This may take longer but provides deeper insights*"
        )
    else:
        st.info(
            "⚡ **Standard Mode Active**\n\n"
            "Fast analysis using pre-built tools. "
            "Switch to Deep Research for complex questions."
        )

    st.markdown("---")

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

    # Tool Category Filter
    with st.expander("🔧 Tools", expanded=False):
        st.markdown("**Select which tools to enable:**")

        tool_categories = get_tool_categories()
        selected_categories = []

        # Create checkboxes for each category (except Data Query)
        for category, description in tool_categories.items():
            # Check if category is currently enabled
            is_enabled = category in st.session_state.enabled_tool_categories

            # Create checkbox with description
            checkbox_value = st.checkbox(
                f"**{category}**",
                value=is_enabled,
                key=f"tool_cat_{category}",
                help=description
            )

            # Show description below checkbox
            if checkbox_value:
                st.caption(f"✓ {description}")
                selected_categories.append(category)
            else:
                st.caption(f"  {description}")

        st.markdown("---")

        # Update session state if selection changed
        if set(selected_categories) != set(st.session_state.enabled_tool_categories):
            st.session_state.enabled_tool_categories = selected_categories

        # Show currently enabled tools count
        total_categories = len(tool_categories)
        enabled_count = len(selected_categories)
        st.caption(f"📊 **{enabled_count}/{total_categories} optional tool categories enabled**")


    st.markdown("---")

    # Conversation status
    num_messages = len(st.session_state.messages)
    num_exchanges = num_messages // 2

    if num_messages > 0:
        if num_messages >= MAX_MESSAGES:
            st.warning(f"⚠️ Conversation history at limit ({num_exchanges} exchanges). Older messages are being removed to maintain performance.")
        else:
            st.info(f"💬 Current conversation: {num_exchanges} exchanges ({num_messages} messages)")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.current_visualizations = []
        st.session_state.agent_logs = ""
        clean_old_visualizations()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")

    # Deep Research examples
    if st.session_state.deep_research_mode:
        with st.expander("🔬 Deep Research Examples", expanded=True):
            st.markdown("""
            **Complex Multi-Step Analysis:**

            - Recommend 5 stocks to long and 5 stocks to short based on fundamental analysis, momentum, and risk-adjusted returns
            - Design and backtest a momentum-based trading strategy for DJ30 stocks from 2020-2023
            - Identify the optimal portfolio allocation across sectors to maximize Sharpe ratio while limiting drawdown
            - Analyze the relationship between Fed rate changes and tech stock performance, including lag effects
            - Find companies with improving fundamentals but declining stock prices (value opportunities)
            - Create a risk parity portfolio using macroeconomic regime analysis
            - Implement a pairs trading strategy for correlated stocks in the same sector
            - Analyze the predictive power of earnings call sentiment on stock returns
            """)
            st.caption("💡 Deep Research mode will decompose these into steps and write custom code")

    with st.expander("📊 Market Analysis"):
        st.markdown("""
        - Compare performance of S&P 500, Gold, and Bitcoin from 2020 to 2023
        - What was the maximum drawdown during the 2008 financial crisis?
        - Show me S&P 500 with 50 and 200-day moving averages
        - Calculate monthly returns for Bitcoin in 2021
        """)

    with st.expander("📈 Economic Analysis"):
        st.markdown("""
        - What are the year-over-year inflation trends from 2020 to 2023?
        - Create a dashboard showing unemployment, inflation, and retail sales during COVID
        - How much did the unemployment rate change from 2019 to 2021?
        - Show the relationship between oil prices and inflation
        """)

    with st.expander("⚠️ Risk & Volatility"):
        st.markdown("""
        - Why was the S&P 500 so volatile in March 2020?
        - Explain the volatility spike in oil prices during 2008
        - What caused Bitcoin's extreme volatility in 2021?
        - What indicators moved together during the 2008 crisis?
        - Identify correlated movements on March 11, 2020
        - What was the volatility of the S&P 500 during March 2020?
        - Show me the drawdown chart for Bitcoin from 2021 to 2022
        - Find the most volatile periods for oil prices
        - Analyze drawdowns and recovery time for the stock market
        """)

    with st.expander("🔗 Correlations & Relationships"):
        st.markdown("""
        - Show me the correlation between VIX and S&P 500
        - Create a scatter plot of unemployment vs stock market performance
        - What's the correlation between gold, Bitcoin, and stocks?
        - Analyze the relationship between interest rates and inflation
        """)

    with st.expander("📈 Portfolio Recommendations"):
        st.markdown("""
        - Recommend a balanced long/short portfolio of 5 stocks each
        - Which companies should I long based on value strategy?
        - Generate growth-focused portfolio recommendations
        - What are the best quality companies to invest in right now?
        - Compare portfolio recommendations: value vs growth strategies
        """)

    with st.expander("📊 DJ30 Portfolio Strategies"):
        st.markdown("""
        - Create a portfolio going long 5 most volatile stocks and short 5 least volatile stocks in the past year
        - Build a momentum-based portfolio with top 5 gainers and bottom 5 losers from 2023
        - Construct a sector-diversified portfolio with best stock from each sector
        - Which DJ30 stocks had the highest volatility in 2024?
        - Show me performance comparison of all tech stocks in DJ30 from 2020-2024
        """)

    with st.expander("🎯 GMV Portfolio Analysis", expanded=False):
        st.markdown("""
        **Portfolio Construction:**
        - Construct a Global Minimum Variance portfolio for DJ30 stocks from 2020 to 2022 and visualize the efficient frontier
        - Build a GMV portfolio using only technology stocks (AAPL, MSFT, CRM, etc.) and visualize the efficient frontier
        - What are the optimal GMV weights for a portfolio of AAPL, MSFT, JPM, and JNJ?
        - Build a minimum variance portfolio and show me the largest position allocations

        **Backtesting & Evaluation:**
        - Create a GMV portfolio trained on 2020-2021 data and evaluate it on 2022-2023
        - Evaluate a GMV portfolio on Q1 2023 data and report the annualized Sharpe ratio
        - Compare the Sharpe ratio of a GMV portfolio vs equal-weight portfolio

        **Advanced Analysis:**
        - Build a GMV portfolio for Q1 2020, then backtest it on Q2-Q4 2020 to see how it performed during COVID
        - Compare GMV portfolio performance across different market regimes (2019 pre-COVID, 2020 crisis, 2021-2022 recovery)
        """)


    with st.expander("💹 DJ30 Stock Analysis"):
        st.markdown("""
        - Show me a candlestick chart for AAPL from 2023 to 2024
        - Compare price performance of AAPL, MSFT, and GOOGL over the past year
        - What was the volatility of AAPL during 2020?
        - Analyze returns for MSFT from 2020 to 2023
        - Create a volatility chart for AAPL showing rolling 30-day volatility
        - What was the price range for JPM in 2022?
        """)

    with st.expander("💼 Company Fundamentals"):
        st.markdown("""
        - What are the fundamentals for AAPL?
        - Compare AAPL, MSFT, and GOOGL on ROE, P/E ratio, and EPS growth
        - Find all companies with ROE above 20% and P/E below 20
        - Show me AAPL's EPS and ROE evolution from 2015 to 2023
        - Create a scatter plot of ROE vs P/E ratio for all companies
        - How does AAPL's ROE correlate with the Fed Funds rate?
        """)

    with st.expander("📰 News & Events"):
        st.markdown("""
        - What were the most popular news on January 22nd, 2012?
        - Show me headlines from the 2008 financial crisis
        - What major events affected the market during the COVID pandemic?
        - Create a timeline of significant market events from 2020-2022
        """)

# Agent Logs Sidebar (Left) - Always visible
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 🔍 Agent Thought Process")
    st.caption("View the agent's reasoning and tool calls in real-time")

    if st.session_state.agent_logs:
        # Show logs in expandable section
        with st.expander("📜 View Agent Logs", expanded=st.session_state.show_logs):
            st.code(st.session_state.agent_logs, language="text")

    else:
        st.info("💡 Agent logs will appear here once you ask a question. You'll be able to see the agent's tool usage, reasoning process, and decision-making in real-time!")

# Main content
st.markdown('<div class="main-header">Financial Data Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about macroeconomic data, market factors, and company fundamentals from 2008 to present</div>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    msg_type = message.get("type", None)

    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>',
                   unsafe_allow_html=True)
    elif role == "system":
        # Handle system messages (routing, planning)
        if msg_type == "routing":
            # Display routing decision as a status widget
            route = message.get("route", "unknown")
            icon = "🔬" if route == "deep_research" else "⚡"
            with st.status(f"{icon} {content}", state="complete", expanded=False):
                st.write(f"Query classified as: **{route.replace('_', ' ').title()}**")
        elif msg_type == "plan":
            # Display plan as an expander
            with st.expander("🧠 Research Plan", expanded=False):
                st.markdown(content)
    else:
        # Use container for styling but render markdown properly
        st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
        st.markdown("**🤖 Analyst:**")
        st.markdown(content)  # This will properly render markdown including ## headers
        st.markdown('</div>', unsafe_allow_html=True)

        # Display visualizations for this message
        if "visualizations" in message:
            for viz_id in message["visualizations"]:
                load_visualization(viz_id)

# User input
with st.container():
    # If clear_input flag is set, reset the text area
    if st.session_state.clear_input:
        st.session_state.user_input = ""
        st.session_state.clear_input = False

    user_input = st.text_area(
        "Ask your question:",
        placeholder="E.g., What was the S&P 500 volatility during the 2008 crisis?",
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

    # Capture start time for visualization tracking
    generation_start_time = datetime.now().timestamp()

    # Truncate conversation history to prevent context overflow
    truncate_conversation_history()

    # Clear current visualizations (will be repopulated after extraction)
    st.session_state.current_visualizations = []


    # Determine mode via Router Agent (if Auto-Routing is on)
    if "auto_routing" not in st.session_state:
        st.session_state.auto_routing = True

    # Logic: If Auto-Routing is ON, use classifier. Else use the manual toggle state.
    use_deep_research = st.session_state.deep_research_mode # Initialize with current manual toggle state

    routing_message = None

    if st.session_state.auto_routing:
        with st.status("🔄 Routing query...", expanded=False) as status:
            from agents.router_agent import classify_query
            route = classify_query(user_input) # Use user_input here
            status.write(f"Classified as: **{route.upper()}**")

            if "complex" in route:
                use_deep_research = True
                routing_label = "✅ Routed to Deep Research Agent"
                routing_message = {
                    "role": "system",
                    "type": "routing",
                    "content": routing_label,
                    "route": "deep_research"
                }
                status.update(label=routing_label, state="complete", expanded=False)
            else:
                use_deep_research = False
                routing_label = "✅ Routed to Financial Analyst"
                routing_message = {
                    "role": "system",
                    "type": "routing",
                    "content": routing_label,
                    "route": "financial_analyst"
                }
                status.update(label=routing_label, state="complete", expanded=False)

            # Update the toggle state visually to match decision
            st.session_state.deep_research_mode = use_deep_research

        # Add routing message to history
        if routing_message:
            st.session_state.messages.append(routing_message)
    else:
        # If auto-routing is off, use the manual toggle state directly
        use_deep_research = st.session_state.deep_research_mode


    # Show loading state with appropriate message
    # spinner_message = "🔬 Conducting deep research..." if use_deep_research else "Thinking..."

    # with st.spinner(spinner_message):
    try:
        # Choose analysis mode based on routing decision
        if use_deep_research:
            # Deep Research Mode: Multi-step analysis with code execution

            # Create a placeholder for the plan in the main chat area
            plan_placeholder = st.empty()

            # Pass plan_placeholder to update UI in real-time
            response = run_deep_research_with_logs(user_input, plan_placeholder=plan_placeholder)

            # Store the plan as a system message for persistence
            if st.session_state.current_plan:
                st.session_state.messages.append({
                    "role": "system",
                    "type": "plan",
                    "content": st.session_state.current_plan
                })
        else:
            # Standard Mode: Fast analysis with pre-built tools
            response = run_analysis_with_logs(
                user_input,
                st.session_state.messages,
                enabled_tool_categories=st.session_state.enabled_tool_categories
            )

        # Validate response - check for empty or incomplete responses
        response_cleaned = response.strip() if response else ""

        # Check for invalid responses
        is_invalid = False
        error_msg = ""

        if not response_cleaned or response_cleaned in ["```", "``", "`"]:
            is_invalid = True
            error_msg = (
                "⚠️ The agent returned an incomplete response. This usually happens due to:\n"
                "1. Context overflow (try clearing conversation history)\n"
                "2. Tool output being too large\n"
                "3. LLM formatting confusion\n\n"
                "Please try:\n"
                "- Simplifying your question\n"
                "- Clearing conversation history\n"
                "- Asking again"
            )

        # Deep Research mode: Check if agent returned planning instead of final report
        elif st.session_state.deep_research_mode:
            # Detect planning/intermediate output patterns
            planning_indicators = [
                "here's my plan",
                "here's a breakdown",
                "i will proceed",
                "let me start by",
                "i'll start with",
                "step 1:",
                "first, i'll",
                "first, i need to",
            ]

            response_lower = response_cleaned[:500].lower()  # Check first 500 chars

            # Check if response looks like planning rather than final report
            has_planning_language = any(indicator in response_lower for indicator in planning_indicators)
            lacks_markdown_header = not response_cleaned.startswith("#")

            # Allow if it has markdown structure even with planning language
            has_proper_structure = "## executive summary" in response_lower or "## methodology" in response_lower

            if has_planning_language and lacks_markdown_header and not has_proper_structure:
                is_invalid = True
                error_msg = (
                    "⚠️ **Deep Research Error: Agent returned planning instead of analysis**\n\n"
                    "The agent provided its workflow/plan instead of executing code and analyzing data.\n\n"
                    "**What happened:** The agent stopped after the planning phase without running any code.\n\n"
                    "**Please try:**\n"
                    "1. Click 'Ask' again - the agent should execute code on the next attempt\n"
                    "2. Simplify your question slightly\n"
                    "3. Clear conversation history if the issue persists\n\n"
                    "💡 **Tip:** Deep Research mode requires the agent to execute Python code. "
                    "If this keeps happening, try Standard Mode instead."
                )

        if is_invalid:
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "visualizations": []
            })
        else:
            # Extract visualization IDs from response using the start time
            viz_ids = extract_visualization_ids(response, start_time=generation_start_time)
            st.session_state.current_visualizations = viz_ids

            # Clean up old visualization files created before this query
            viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
            if os.path.exists(viz_dir):
                for viz_file in glob.glob(os.path.join(viz_dir, "viz_*.json")):
                    try:
                        mtime = os.path.getmtime(viz_file)
                        # Delete files created BEFORE this query started
                        if mtime < generation_start_time:
                            viz_id = os.path.basename(viz_file).replace('.json', '')
                            # Only delete if not referenced in conversation history
                            is_referenced = any(
                                viz_id in msg.get("visualizations", [])
                                for msg in st.session_state.messages
                            )
                            if not is_referenced:
                                os.remove(viz_file)
                    except (OSError, Exception):
                        continue

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "visualizations": viz_ids
            })


    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            "visualizations": []
        })

    # Set flag to clear input on next run
    st.session_state.clear_input = True

    # Rerun to display new messages
    st.rerun()


