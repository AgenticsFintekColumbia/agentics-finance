"""
GMV (Global Minimum Variance) Portfolio Construction Tools for DJ30.
Uses nodewise Lasso regression to estimate precision matrix and compute optimal portfolio weights.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import json
from tools.nodewise_gmv_tool import (
    nodewise_gmv_from_csv,
    evaluate_portfolio_from_weights
)


class GMVPortfolioConstructionInput(BaseModel):
    """Input for GMV Portfolio Construction Tool."""
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format for training period")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format for training period")
    tickers: Optional[str] = Field(None, description="Optional comma-separated list of ticker symbols to restrict portfolio (e.g., 'AAPL,MSFT,JPM'). If not provided, uses all available DJ30 stocks.")


class GMVPortfolioConstructionTool(BaseTool):
    name: str = "Construct GMV Portfolio"
    description: str = (
        "Constructs a Global Minimum Variance (GMV) portfolio for DJ30 stocks using nodewise Lasso regression. "
        "This tool estimates the precision (inverse covariance) matrix via nodewise regression and computes "
        "optimal portfolio weights that minimize portfolio variance. "
        "Uses DJ30 stock data automatically (data/dj30_data_full.csv). "
        "Use this to build low-risk, diversified portfolios based on historical excess returns. "
        "Returns portfolio weights that sum to 1.0, along with metadata about the construction."
    )
    args_schema: type[BaseModel] = GMVPortfolioConstructionInput

    def _run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[str] = None
    ) -> str:
        try:
            # Parse tickers if provided
            ticker_list = None
            if tickers:
                ticker_list = [t.strip().upper() for t in tickers.split(',')]

            # Compute GMV weights using nodewise regression
            # csv_path defaults to "data/dj30_data_full.csv" in the function
            weights_series = nodewise_gmv_from_csv(
                start_date=start_date,
                end_date=end_date,
                tickers=ticker_list
            )

            # Convert to dictionary
            weights_dict = weights_series.to_dict()

            # Calculate summary statistics
            num_assets = len(weights_dict)
            max_weight_ticker = max(weights_dict.items(), key=lambda x: x[1])
            min_weight_ticker = min(weights_dict.items(), key=lambda x: x[1])

            # Sort weights for better readability
            sorted_weights = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True))

            result = {
                "success": True,
                "weights": sorted_weights,
                "metadata": {
                    "data_source": "data/dj30_data_full.csv",
                    "start_date": start_date or "earliest available",
                    "end_date": end_date or "latest available",
                    "num_assets": num_assets,
                    "tickers": list(sorted_weights.keys()),
                    "max_weight": {
                        "ticker": max_weight_ticker[0],
                        "weight": float(max_weight_ticker[1])
                    },
                    "min_weight": {
                        "ticker": min_weight_ticker[0],
                        "weight": float(min_weight_ticker[1])
                    }
                },
                "summary": (
                    f"GMV Portfolio constructed with {num_assets} assets from {start_date or 'earliest'} to {end_date or 'latest'}.\n"
                    f"  Largest allocation: {max_weight_ticker[0]} ({max_weight_ticker[1]*100:.2f}%)\n"
                    f"  Smallest allocation: {min_weight_ticker[0]} ({min_weight_ticker[1]*100:.2f}%)\n"
                    f"  All weights sum to 100%"
                )
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class PortfolioEvaluationInput(BaseModel):
    """Input for Portfolio Evaluation Tool."""
    weights: str = Field(..., description="JSON string of portfolio weights as dictionary {ticker: weight}, e.g., '{\"AAPL\": 0.5, \"MSFT\": 0.5}'")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format for evaluation period")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format for evaluation period")


class PortfolioEvaluationTool(BaseTool):
    name: str = "Evaluate Portfolio Performance"
    description: str = (
        "Evaluates portfolio performance on a specified date range (typically a holdout/test period). "
        "Given portfolio weights, this tool computes realized excess return statistics including: "
        "daily and annualized mean return, variance, standard deviation (volatility), and Sharpe ratio. "
        "Uses DJ30 stock data automatically (data/dj30_data_full.csv). "
        "Use this to backtest portfolio performance or validate portfolio construction strategies. "
        "The evaluation uses excess returns (returns above the risk-free rate)."
    )
    args_schema: type[BaseModel] = PortfolioEvaluationInput

    def _run(
        self,
        weights: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        try:
            # Parse weights from JSON string
            try:
                weights_dict = json.loads(weights)
            except json.JSONDecodeError:
                return json.dumps({
                    "success": False,
                    "error": "Invalid weights format. Must be a JSON dictionary like {\"AAPL\": 0.5, \"MSFT\": 0.5}"
                })

            # Evaluate portfolio
            # csv_path defaults to "data/dj30_data_full.csv" in the function
            stats = evaluate_portfolio_from_weights(
                weights=weights_dict,
                start_date=start_date,
                end_date=end_date
            )

            # Format result
            result = {
                "success": True,
                "evaluation_period": {
                    "data_source": "data/dj30_data_full.csv",
                    "start_date": start_date or "earliest available",
                    "end_date": end_date or "latest available",
                    "n_observations": stats["n_obs"]
                },
                "performance_metrics": {
                    "daily": {
                        "mean_return": stats["mean_daily"],
                        "variance": stats["var_daily"],
                        "volatility": stats["std_daily"],
                        "sharpe_ratio": stats["sharpe_daily"]
                    },
                    "annualized": {
                        "mean_return": stats["mean_annualized"],
                        "variance": stats["var_annualized"],
                        "volatility": stats["std_annualized"],
                        "sharpe_ratio": stats["sharpe_annualized"]
                    }
                },
                "summary": (
                    f"Portfolio Performance ({stats['n_obs']} observations):\n"
                    f"  Annualized Return: {stats['mean_annualized']*100:.2f}%\n"
                    f"  Annualized Volatility: {stats['std_annualized']*100:.2f}%\n"
                    f"  Sharpe Ratio: {stats['sharpe_annualized']:.4f}\n"
                    f"\n"
                    f"Daily metrics:\n"
                    f"  Mean Return: {stats['mean_daily']*100:.4f}%\n"
                    f"  Volatility: {stats['std_daily']*100:.4f}%\n"
                    f"  Sharpe Ratio: {stats['sharpe_daily']:.4f}"
                )
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })
