import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from portfolio_analysis.scripts.data.portfolio import PortfolioAnalysis, TickerAnalysis, calculate_kpis
from portfolio_analysis.scripts.visualization.plots import create_pie_chart, create_bar_chart, \
    create_ticker_performance_chart, create_transaction_annotated_line_chart


@st.cache_data
def read_data(tickers, start_date, end_date):
    ticker_analysis = TickerAnalysis(tickers, start_date, end_date)
    ticker_data = ticker_analysis.fetch_data()

    # Analyze portfolio
    portfolio_analysis = PortfolioAnalysis(transactions, ticker_data)
    portfolio_performance, portfolio_ticker_performance = portfolio_analysis.calculate_portfolio_performance()
    allocation_df = pd.DataFrame()
    for ticker, value in portfolio_ticker_performance.items():
        allocation_df = pd.concat((allocation_df, portfolio_ticker_performance[ticker].iloc[-1:]))
    allocation_df = allocation_df[['Ticker', 'Quantity Held', 'Market Value', 'Realized Gains', 'Unrealized Gains']]
    allocation_df = allocation_df.groupby('Ticker').sum().reset_index()
    portfolio_kpis = calculate_kpis(portfolio_performance,
                                    portfolio_ticker_performance,
                                    allocation_df)

    return portfolio_kpis, allocation_df, portfolio_performance, ticker_data



st.set_page_config(
    page_title="Portfolio Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File uploader for transactions
# uploaded_file = st.file_uploader("Upload your CSV file with transactions", type="csv")
uploaded_file = True

if uploaded_file:
    # Load transactions
    # transactions = pd.read_csv(uploaded_file, parse_dates=['Date'])
    transactions = pd.read_csv('./tests/transactions.csv',
                               parse_dates=['Date'])
    transactions['Date'] = pd.to_datetime(transactions['Date'], format="%d/%m/%Y")

    # Prepare data for analysis
    tickers = transactions['Ticker'].unique().tolist()
    start_date = transactions['Date'].min()
    end_date = datetime.today().date()

    # Fetch historical data
    #st.write("Fetching historical price data...")
    # ticker_analysis = TickerAnalysis(tickers, start_date, end_date)
    # ticker_data = ticker_analysis.fetch_data()
    #
    # # Analyze portfolio
    # portfolio_analysis = PortfolioAnalysis(transactions, ticker_data)
    # portfolio_performance, portfolio_ticker_performance = portfolio_analysis.calculate_portfolio_performance()
    # allocation_df = pd.DataFrame()
    # for ticker, value in portfolio_ticker_performance.items():
    #     allocation_df = pd.concat((allocation_df, portfolio_ticker_performance[ticker].iloc[-1:]))
    # allocation_df = allocation_df[['Ticker', 'Quantity Held', 'Market Value', 'Realized Gains', 'Unrealized Gains']]
    # allocation_df = allocation_df.groupby('Ticker').sum().reset_index()
    # portfolio_kpis = calculate_kpis(portfolio_performance,
    #                                 portfolio_ticker_performance,
    #                                 allocation_df)

    portfolio_kpis, allocation_df, portfolio_performance, ticker_data = read_data(tickers,
                                                                                  start_date,
                                                                                  end_date)

    # menu = st.sidebar.radio("Navigation", ["Home", "About", "Contact"])
    # if menu == "Home":
    #     st.header("Welcome to the Home Page")
    #     st.write("This is the main content.")
    # elif menu == "About":
    #     st.header("About Us")
    #     st.write("Details about the app or company.")
    # elif menu == "Contact":
    #     st.header("Contact Us")
    #     st.write("Email: contact@example.com")
    st.markdown(
        """
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-XXXXXXX-X"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'UA-XXXXXXX-X');
        </script>
        """,
        unsafe_allow_html=True,
    )
    # st.sidebar.header("Contact Us")
    # st.sidebar.write("Email: example@domain.com")
    # st.sidebar.write("Phone: +123456789")

    # tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Settings"])
    # with tab1:
    #     st.write("Overview content")
    # with tab2:
    #     st.write("Analysis content")
    # with tab3:
    #     st.write("Settings content")

    # st.markdown("""
    #     <style>
    #     :root {
    #         --primary-color: #007BFF;
    #         --secondary-color: #6C757D;
    #         --accent-color: #28A745;
    #         --bg-color: #F8F9FA;
    #         --text-color: #212529;
    #     }
    #     .header {
    #         color: var(--primary-color);
    #         font-size: 24px;
    #         font-weight: bold;
    #     }
    #     </style>
    # """, unsafe_allow_html=True)

    # st.markdown("""
    #     <style>
    #     .hover-effect:hover {
    #         color: var(--accent-color);
    #         cursor: pointer;
    #         transform: scale(1.05);
    #     }
    #     </style>
    # """, unsafe_allow_html=True)
    #
    # st.markdown("<div class='hover-effect'>Click here for more info</div>", unsafe_allow_html=True)

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        body {
            font-family: 'Helvetica', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

    # st.markdown("""
    #     <div style="background: linear-gradient(to right, #4CAF50, #81C784); padding: 10px; border-radius: 5px;">
    #         <h2 style="color: white;">Portfolio Value</h2>
    #     </div>
    # """, unsafe_allow_html=True)


    # dark_mode = st_toggle_switch("Dark Mode", default_value=False)
    # if dark_mode:
    #     st.markdown("<style>body {background-color: #121212; color: white;}</style>", unsafe_allow_html=True)
    st.success("Portfolio loaded successfully!")

    st.title('Your Porftolio Analysis')
    st.subheader("Detailed insights into your financial portfolio")
    st.markdown("<hr style='border: 1px solid blue;'>", unsafe_allow_html=True)

    # col1, col2 = st.columns(2)
    col1, spacer, col2 = st.columns([2, 0.2, 1])  # Adjust ratios for design

    with col1:
        # st.markdown(f"""
        #     <div style="display: flex; gap: 20px;">
        #         <div style="background-color: #111111; padding: 15px; border-radius: 5px; flex: 1;">
        #             <h5>🚀 Best Performing Ticker</h5>
        #             <h4>{portfolio_kpis['best_ticker']['ticker']}</h4>
        #         </div>
        #         <div style="background-color: #111111; padding: 15px; border-radius: 5px; flex: 1;">
        #             <h5>📉 Worst Performing Ticker</h5>
        #             <h4>{portfolio_kpis['worst_ticker']['ticker']}</h4>
        #         </div>
        #     </div>
        # """, unsafe_allow_html=True)
        with st.container():
            sub_col1, sub_col2 = st.columns(2, vertical_alignment='center')
            with sub_col1:
                st.metric(label="Best Performing Ticker", value=portfolio_kpis['best_ticker']['ticker'])
                st.metric(label="Value", value=f"€ {portfolio_kpis['best_ticker']['unrealized_gains']:.0f}")
                st.metric(label="Performance",
                          value=f"{portfolio_kpis['best_ticker']['performance']:.0f} %",
                          delta_color="inverse"
                          )

            with sub_col2:
                st.metric(label="Worst Performing Ticker", value=portfolio_kpis['worst_ticker']['ticker'])
                st.metric(label="Value", value=f"€ {portfolio_kpis['worst_ticker']['unrealized_gains']:.0f}")
                st.metric(label="Performance",
                          value=f"{portfolio_kpis['worst_ticker']['performance']:.0f} %",
                          delta_color="inverse")
            st.markdown('<div style="margin-bottom: 50px;"></div>', unsafe_allow_html=True)
            st.markdown("---")

        with st.container():
            st.header("Portfolio Value with Transactions")
            annotated_line_chart = create_transaction_annotated_line_chart(portfolio_performance, transactions, ticker_data)
            annotated_line_chart.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value (€)",
                template="plotly_white",
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            st.plotly_chart(annotated_line_chart)

        st.markdown("---")
        st.header(f"Performance of selected ticker")
        st.info("Select a ticker to view its performance over time.")
        selected_ticker = st.selectbox("", tickers)
        st.subheader(f"{selected_ticker.upper()}")
        ticker_chart = create_ticker_performance_chart(selected_ticker, ticker_data, transactions)
        st.plotly_chart(ticker_chart)

    with spacer:
        st.markdown("<div style='height:100%; border-left:1px solid #CCC;'></div>", unsafe_allow_html=True)

    with col2:
        st.metric(label="Portfolio Total Value", value=f"€ {portfolio_kpis['value']:,.0f}")
        st.metric(label="Portfolio Net", value=f"€ {portfolio_kpis['unrealized_gains']:.2f}")
        st.metric(label="Portfolio Performance",
                  value=f"{portfolio_kpis['performance']:.0f} %",
                  delta_color="inverse",
                  help="Percentage change in the total portfolio value compared to the initial value"
                  )

        st.markdown('<div style="margin-bottom: 50px;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.header("Current Portfolio Allocation")
        pie_chart = create_pie_chart(allocation_df)
        st.plotly_chart(pie_chart)

        st.markdown("---")
        st.header("Portfolio Value")
        bar_chart = create_bar_chart(allocation_df)
        st.plotly_chart(bar_chart)

    st.markdown("---")
    st.write("Transactions preview:")
    st.dataframe(transactions, use_container_width=True)

    st.markdown("""
            <footer style="text-align: center; margin-top: 50px; font-size: small; color: #666;">
                © 2025 Portfolio Analysis | All Rights Reserved
            </footer>
        """, unsafe_allow_html=True)
