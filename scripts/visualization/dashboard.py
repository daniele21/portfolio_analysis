from typing import Dict

import numpy as np
import pandas as pd
from bokeh.layouts import gridplot, column
from bokeh.models import Tabs, ColumnDataSource, DateRangeSlider

from core.operations.time_series import portfolio_return, portfolio_vol
from core.portfolio.portfolio import Portfolio
from core.portfolio.tickers import Tickers
from scripts.visualization.info import plot_info_table
from scripts.visualization.optimization import optimization_plot
from scripts.visualization.panel import tab_figures
from scripts.visualization.stake import stake_plot
from scripts.visualization.trend import plot_stock_price, plot_ticker_volume, plot_performance


class FinanceDashboard:

    def __init__(self,
                 tickers: Tickers,
                 portfolio: Portfolio):
        self.tickers = tickers
        self.portfolio = portfolio
        self.ticker_ids = self.tickers.get_ticker_ids()

    def ticker_data_plot(self) -> Tabs:
        tabs_dict = {}

        for ticker_id in self.ticker_ids:
            stock = ColumnDataSource(
                data=dict(Date=[], Open=[], Close=[], High=[], Low=[], index=[]))
            ticker = self.tickers.get_ticker(ticker_id)
            ticker.data['Date'] = pd.to_datetime(ticker.data.index)
            ticker.data = ticker.data.reset_index(drop=True)
            stock.data = stock.from_df(ticker.data)
            if len(stock.data['index']) <= 1:
                continue

            ticker_details = self.tickers.get_ticker_details(ticker_id)
            # text_inputs = [{'title': info, 'value': ticker_details[info]} for info in ticker_details]
            text_inputs = {'info': list(ticker_details.keys()),
                           'value': list(ticker_details.values())}

            start_date, end_date = stock.data['index'][0], stock.data['index'][-1]
            date_range_slider = DateRangeSlider(title='Data Range',
                                                value=(start_date, end_date),
                                                start=start_date, end=end_date)

            fig_stock = plot_stock_price(stock, date_range_slider)
            fig_volume = plot_ticker_volume(stock, date_range_slider)
            fig_info = plot_info_table(text_inputs)

            fig_data = column([fig_stock, fig_volume])

            # fig = gridplot([[date_range_slider, None], [fig_data, fig_info]],
            #                merge_tools=True)
            fig = gridplot([[fig_data, fig_info]],
                           merge_tools=True)
            tabs_dict[ticker_id] = fig

        tabs = tab_figures(tabs_dict)

        return tabs

    def ticker_performance_plot(self) -> Tabs:
        tabs_dict = {'Portfolio': None}
        for instr in self.tickers.instruments:
            tabs_dict[instr] = None

        for ticker_id in self.ticker_ids:
            ticker = self.tickers.get_ticker(ticker_id)
            ticker.data['Date'] = pd.to_datetime(ticker.data.index)
            ticker.data = ticker.data.reset_index(drop=True)

            performance_df = self.portfolio.get_ticker_performance(ticker).copy(True)
            performance_df = performance_df.reset_index()[['Date', 'performance']]

            stock = ColumnDataSource(
                data=dict(Date=[], performance=[], index=[]))
            stock.data = stock.from_df(performance_df)
            if len(stock.data['index']) <= 1:
                continue

            start_date, end_date = stock.data['index'][0], stock.data['index'][-1]
            date_range_slider = DateRangeSlider(title='Data Range',
                                                value=(start_date, end_date),
                                                start=start_date, end=end_date)
            fig_perf = plot_performance(stock, date_range_slider)

            ticker_details = self.tickers.get_ticker_details(ticker_id)
            text_inputs = {'info': list(ticker_details.keys()),
                           'value': list(ticker_details.values())}
            fig_info = plot_info_table(text_inputs)

            # fig = gridplot([[date_range_slider, None], [fig_perf, fig_info]],
            #                merge_tools=True)
            fig = gridplot([[fig_perf, fig_info]],
                           merge_tools=True)
            tabs_dict[ticker_id] = fig

        portfolio_performance = self.portfolio.get_portfolio_performance(self.tickers)
        performance_df = portfolio_performance.reset_index()[['Date', 'performance']]
        stock = ColumnDataSource(
            data=dict(Date=[], performance=[], index=[]))
        stock.data = stock.from_df(performance_df)
        start_date, end_date = stock.data['index'][0], stock.data['index'][-1]
        date_range_slider = DateRangeSlider(title='Data Range',
                                            value=(start_date, end_date),
                                            start=start_date, end=end_date)
        fig_perf = plot_performance(stock, date_range_slider)

        text_inputs = {'info': ['Portfolio Performance'],
                       'value': [performance_df['performance'][0]]}
        fig_info = plot_info_table(text_inputs)

        # fig = gridplot([[date_range_slider, None], [fig_perf, fig_info]],
        #                merge_tools=True)
        fig = gridplot([[fig_perf, fig_info]],
                       merge_tools=True)

        tabs_dict['Portfolio'] = fig

        for group, df in self.portfolio.get_group_performances(tickers=self.tickers).items():
            fig = self._performance_plot(df, text_inputs={'info': [f'{group} performance'],
                                                    'value': [df['performance'][-1]]})
            tabs_dict[group] = fig

        tabs = tab_figures(tabs_dict)

        return tabs

    def stake_status_plot(self):
        instrument_stake = {}
        stake = self.portfolio.get_actual_stake(self.tickers)
        instrument_stake['General'] = stake

        for instr in self.tickers.instruments:
            instr_stake = self.portfolio.get_actual_stake_by_instrument(instrument=instr,
                                                                        tickers=self.tickers)
            instrument_stake[instr] = instr_stake

        etf_risk_stake = self.portfolio.get_actual_stake_by_risk(tickers=self.tickers)
        instrument_stake['Risk'] = etf_risk_stake

        tabs_dict = {}
        for title, stake in instrument_stake.items():
            stake_fig = stake_plot(stake, title=title)
            tabs_dict[title] = stake_fig

        tabs = tab_figures(tabs_dict)

        return tabs

    def portfolio_optimization(self):
        ticker_group = {'All': None}
        ticker_group.update({instr: self.tickers.ticker_by_instr[instr] for instr in self.tickers.instruments})
        group_tabs = {}

        for group in ticker_group:
            ef, er, cov = self.tickers.get_efficient_frontier(n_points=30,
                                                              freq='M',
                                                              periods_per_year=12,
                                                              features=ticker_group[group])
            rets = [portfolio_return(w, er) for w in ef['weights']]
            vols = [portfolio_vol(w, cov) for w in ef['weights']]
            ef = pd.DataFrame({
                "returns": rets,
                "volatility": vols,
                "weights": ef['weights']
            })

            ticker_weights = np.array(ef['weights'].to_list())
            ticker_weights_df = pd.DataFrame(ticker_weights, columns=cov.columns)

            ef = pd.concat((ef.reset_index(drop=True), ticker_weights_df), axis=1)
            ef = ef.set_index('volatility')

            fig = optimization_plot(ef, er, cov, title=f'{group} optimization')
            group_tabs[group] = fig

        tabs = tab_figures(group_tabs)

        return tabs

    def _performance_plot(self, df: pd.DataFrame, text_inputs: Dict):
        performance_df = df.reset_index()[['Date', 'performance']]
        stock = ColumnDataSource(
            data=dict(Date=[], performance=[], index=[]))
        stock.data = stock.from_df(performance_df)
        start_date, end_date = stock.data['index'][0], stock.data['index'][-1]
        date_range_slider = DateRangeSlider(title='Data Range',
                                            value=(start_date, end_date),
                                            start=start_date, end=end_date)
        fig_perf = plot_performance(stock, date_range_slider)

        # text_inputs = {'info': ['Portfolio Performance'],
        #                'value': [performance_df['performance'][0]]}
        fig_info = plot_info_table(text_inputs)

        fig = gridplot([[date_range_slider, None], [fig_perf, fig_info]],
                       merge_tools=True)

        return fig
