# finance_dashboard/bokeh_app.py

from bokeh.embed import components
from portfolio_analysis.scripts.visualization.dashboard import FinanceDashboard
from config import TICKERS_PATH, TRANSACTIONS_PATH, TICKER_DATA_DIR
from portfolio_analysis.core.portfolio.portfolio import Portfolio
from portfolio_analysis.core.portfolio.tickers import Tickers

def create_dashboard():
    # Inizializza i tuoi oggetti Portfolio e Tickers
    tickers = Tickers(TICKERS_PATH, TICKER_DATA_DIR)
    portfolio = Portfolio(TRANSACTIONS_PATH)

    # Crea l'istanza del dashboard
    dashboard = FinanceDashboard(tickers, portfolio)

    # Crea i vari plot
    stake_plot = dashboard.stake_status_plot()
    history_plot = dashboard.ticker_data_plot()
    performance_plot = dashboard.ticker_performance_plot()
    optimization_plot = dashboard.portfolio_optimization()

    # Imposta sizing_mode per ogni grafico
    plots = {
        'stake': {
            'script': '',
            'div': '',
            'title': 'Stake',
            'plot': stake_plot
        },
        'history': {
            'script': '',
            'div': '',
            'title': 'History',
            'plot': history_plot
        },
        'performance': {
            'script': '',
            'div': '',
            'title': 'Performance',
            'plot': performance_plot
        },
        'optimization': {
            'script': '',
            'div': '',
            'title': 'Optimization',
            'plot': optimization_plot
        }
    }

    # Imposta sizing_mode e genera script/div per ogni plot
    for key in plots:
        plots[key]['plot'].sizing_mode = "stretch_both"  # Adattamento responsivo sia in larghezza che in altezza
        script, div = components(plots[key]['plot'])
        plots[key]['script'] = script
        plots[key]['div'] = div

    return plots
