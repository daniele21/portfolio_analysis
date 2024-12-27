# app.py
from bokeh.io import curdoc
from bokeh.layouts import layout
from portfolio_analysis.core.portfolio.portfolio import Portfolio
from portfolio_analysis.core.portfolio.tickers import Tickers
from portfolio_analysis.scripts.constants.paths import TICKER_DATA_DIR, DATA_DIR
from portfolio_analysis.scripts.utils.os_manager import ensure_folder
from portfolio_analysis.scripts.visualization.dashboard import FinanceDashboard
from portfolio_analysis.scripts.visualization.panel import tab_figures
from portfolio_analysis.scripts.utils.logging import setup_logger

# Configura il logger
logger = setup_logger('App')

# Percorsi ai file di configurazione
TICKERS_PATH = 'tickers_test.json'          # Sostituisci con il percorso corretto
TRANSACTIONS_PATH = 'transactions_test.json'  # Sostituisci con il percorso corretto

# Assicurati che le cartelle necessarie esistano
ensure_folder(DATA_DIR)
ensure_folder(TICKER_DATA_DIR)

# Inizializza i tuoi oggetti Portfolio e Tickers
tickers = Tickers(TICKERS_PATH, TICKER_DATA_DIR)
portfolio = Portfolio(TRANSACTIONS_PATH)

# Crea l'istanza del dashboard
dashboard = FinanceDashboard(tickers, portfolio)

# Opzionale: Aggiorna i dati se necessario
# tickers.update_tickers_data()

# Crea i vari tab
fig = tab_figures({
    'Stake': dashboard.stake_status_plot(),
    "History": dashboard.ticker_data_plot(),
    'Performance': dashboard.ticker_performance_plot(),
    "Optimization": dashboard.portfolio_optimization()
})

# Organizza i tab in un layout
main_layout = layout([
    [fig]
])

# Aggiungi il layout al documento corrente
curdoc().add_root(main_layout)
curdoc().title = "Finance Dashboard"

logger.info('Finance Dashboard avviato sul Bokeh Server.')
