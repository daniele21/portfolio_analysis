# app.py
from flask import Flask, render_template
from bokeh.resources import INLINE
from finance_dashboard.bokeh_app import create_dashboard
from config import DATA_DIR, TICKER_DATA_DIR, TICKERS_PATH, TRANSACTIONS_PATH
from portfolio_analysis.core.portfolio.portfolio import Portfolio
from portfolio_analysis.core.portfolio.tickers import Tickers
from portfolio_analysis.scripts.utils.os_manager import ensure_folder
from portfolio_analysis.scripts.utils.logging import setup_logger

app = Flask(__name__)

# Configura il logger
logger = setup_logger('App')

# Assicurati che le cartelle necessarie esistano
ensure_folder(DATA_DIR)
ensure_folder(TICKER_DATA_DIR)

@app.route('/')
def index():
    try:
        # Crea le figure Bokeh
        plots = create_dashboard()

        # Recupera le risorse Bokeh
        bokeh_js = INLINE.render_js()
        bokeh_css = INLINE.render_css()

        return render_template('index.html',
                               bokeh_js=bokeh_js,
                               bokeh_css=bokeh_css,
                               plots=plots)
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione del dashboard: {e}")
        return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
