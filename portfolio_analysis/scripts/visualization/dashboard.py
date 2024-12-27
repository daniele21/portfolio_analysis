from typing import Dict, List
from bokeh.layouts import column, row
from bokeh.models import Div
from bokeh.plotting import output_file, save

import numpy as np
import pandas as pd
from bokeh.layouts import gridplot, column
from bokeh.models import Tabs, ColumnDataSource, DateRangeSlider

from portfolio_analysis.core.operations.time_series import portfolio_return, portfolio_vol
from portfolio_analysis.core.portfolio.portfolio import Portfolio
from portfolio_analysis.core.portfolio.tickers import Tickers
from portfolio_analysis.scripts.visualization.info import plot_info_table
from portfolio_analysis.scripts.visualization.optimization import optimization_plot
from portfolio_analysis.scripts.visualization.panel import tab_figures
from portfolio_analysis.scripts.visualization.stake import stake_plot
from portfolio_analysis.scripts.visualization.trend import plot_performance, \
    plot_stock_with_volume


def prepare_column_data_source(data: pd.DataFrame, fields: List[str]) -> ColumnDataSource:
    """Prepare a ColumnDataSource from a DataFrame."""
    source_data = data[fields].reset_index().rename(columns={"index": "Date"})
    return ColumnDataSource(source_data)

def create_date_range_slider(start, end, title="Data Range") -> DateRangeSlider:
    """Create a reusable DateRangeSlider."""
    return DateRangeSlider(title=title, value=(start, end), start=start, end=end)

def create_grid_plot(figures: List, info_data: Dict) -> gridplot:
    """Create a grid layout with figures and an information table."""
    fig_info = plot_info_table(info_data)
    return gridplot([[column(figures), fig_info]], merge_tools=True)


def process_ticker_data(ticker) -> pd.DataFrame:
    """Process ticker data by resetting the index and adding necessary fields."""
    ticker.data['Date'] = pd.to_datetime(ticker.data.index)
    ticker.data = ticker.data.reset_index(drop=True)
    return ticker.data


def performance_data_to_source(performance_df: pd.DataFrame) -> ColumnDataSource:
    """Convert performance data to a Bokeh ColumnDataSource."""
    performance_df = performance_df.reset_index()[['Date', 'performance']]
    return ColumnDataSource(performance_df)


class FinanceDashboard:

    def __init__(self, tickers: Tickers, portfolio: Portfolio):
        self.tickers = tickers
        self.portfolio = portfolio
        self.ticker_ids = self.tickers.get_ticker_ids()

    def ticker_data_plot(self) -> Tabs:
        tabs_dict = {}

        for ticker_id in self.ticker_ids:
            ticker = self.tickers.get_ticker(ticker_id)
            processed_data = process_ticker_data(ticker)

            # Ensure 'Volume' column exists
            if "Volume" not in processed_data.columns:
                processed_data["Volume"] = 0  # Add default values if missing

            if processed_data.empty:
                continue

            source = prepare_column_data_source(processed_data, ['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])
            start_date, end_date = source.data['Date'][0], source.data['Date'][-1]

            fig_stock_volume = plot_stock_with_volume(source)
            ticker_details = self.tickers.get_ticker_details(ticker_id)

            text_inputs = {'info': list(ticker_details.keys()), 'value': list(ticker_details.values())}

            tabs_dict[ticker_id] = create_grid_plot([fig_stock_volume], text_inputs)

        return tab_figures(tabs_dict)

    def ticker_performance_plot(self) -> Tabs:
        tabs_dict = {}

        # Portfolio-level performance
        portfolio_performance = self.portfolio.get_portfolio_performance(self.tickers)
        portfolio_source = performance_data_to_source(portfolio_performance)
        start_date, end_date = portfolio_source.data['Date'][0], portfolio_source.data['Date'][-1]
        date_slider = create_date_range_slider(start_date, end_date)

        fig_perf = plot_performance(portfolio_source, date_slider)
        text_inputs = {'info': ['Portfolio Performance'], 'value': [portfolio_performance.iloc[0]['performance']]}
        tabs_dict['Portfolio'] = create_grid_plot([fig_perf], text_inputs)

        # Individual ticker performance
        for ticker_id in self.ticker_ids:
            ticker = self.tickers.get_ticker(ticker_id)
            processed_data = process_ticker_data(ticker)

            performance_df = self.portfolio.get_ticker_performance(ticker)
            source = performance_data_to_source(performance_df)

            # Check if 'Date' is not empty
            if len(source.data['Date']) > 0:
                start_date, end_date = source.data['Date'][0], source.data['Date'][-1]
                date_slider = create_date_range_slider(start_date, end_date)
                fig_perf = plot_performance(source, date_slider)

                ticker_details = self.tickers.get_ticker_details(ticker_id)
                text_inputs = {'info': list(ticker_details.keys()), 'value': list(ticker_details.values())}
                tabs_dict[ticker_id] = create_grid_plot([fig_perf], text_inputs)

        return tab_figures(tabs_dict)

    def stake_status_plot(self):
        tabs_dict = {}
        stakes = {
            "General": self.portfolio.get_actual_stake(self.tickers),
            **{instr: self.portfolio.get_actual_stake_by_instrument(instr, self.tickers) for instr in self.tickers.instruments},
            "Risk": self.portfolio.get_actual_stake_by_risk(self.tickers)
        }

        for title, stake_data in stakes.items():
            stake_fig = stake_plot(stake_data, title=title)
            tabs_dict[title] = stake_fig

        return tab_figures(tabs_dict)

    def portfolio_optimization(self):
        tabs_dict = {}
        ticker_groups = {"All": None, **{instr: self.tickers.ticker_by_instr[instr] for instr in self.tickers.instruments}}

        for group, tickers in ticker_groups.items():
            ef, er, cov = self.tickers.get_efficient_frontier(n_points=30, freq='ME', periods_per_year=12, features=tickers)
            rets = [portfolio_return(w, er) for w in ef['weights']]
            vols = [portfolio_vol(w, cov) for w in ef['weights']]

            ef = pd.DataFrame({
                "returns": rets,
                "volatility": vols,
                **{col: np.array([w[i] for w in ef['weights']]) for i, col in enumerate(cov.columns)}
            }).set_index("volatility")

            fig = optimization_plot(ef, er, cov, title=f"{group} Optimization")
            tabs_dict[group] = fig

        return tab_figures(tabs_dict)

        # Aggiungi questa funzione alla classe esistente
    def create_dashboard(self, output_path="dashboard.html"):
        # Intestazione
        header = Div(text="<h1>Finance Dashboard</h1>",
                     style={"text-align": "center", "font-family": "Arial", "color": "#4CAF50"})

        # Grafici
        ticker_data_tabs = self.ticker_data_plot()
        ticker_perf_tabs = self.ticker_performance_plot()
        stake_status_tabs = self.stake_status_plot()
        optimization_tabs = self.portfolio_optimization()

        # Footer
        footer = Div(text="<p style='text-align:center;'>© 2024 Financial Insights</p>", style={"color": "gray"})

        # Layout completo
        layout = column(
            header,
            row(ticker_data_tabs, ticker_perf_tabs),
            row(stake_status_tabs, optimization_tabs),
            footer,
            sizing_mode="stretch_width"
        )

        # Esporta la pagina
        output_file(output_path, title="Finance Dashboard")
        save(layout, title="Finance Dashboard")