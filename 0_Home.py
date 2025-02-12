import streamlit as st


from utils.streamlit_utils import load_upload_action_menu


def configure_page():
    st.set_page_config(
        page_title="My Financial Vision",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS (same as before)
    st.markdown(
        """
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #1E3A8A;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 2rem !important;
            text-align: center;
            padding: 1rem;
            border-radius: 8px;
        }
        
        h2 {
            color: #2563EB;
            font-size: 2rem !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #BFDBFE;
        }
        
        h3 {
            color: #3B82F6;
            font-size: 1.5rem !important;
            font-weight: 500 !important;
            margin-top: 1.5rem !important;
        }
        
        /* Content sections */
        .stMarkdown {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
        }
        
        /* Custom cards */
        .custom-card {
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            border: 1px solid #E5E7EB;
        }
        
        /* Feature cards */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .feature-card {
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            transition: transform 0.2s;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* Timeline styling */
        .timeline {
            border-left: 2px solid #3B82F6;
            padding-left: 1.5rem;
            margin-left: 1rem;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 1.5rem;
        }
        
        /* Values section */
        .values-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .value-card {
            padding: 1.2rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #BAE6FD;
        }
        
        /* Product features */
        .product-feature {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 6px;
            border-left: 4px solid #3B82F6;
        }
        
        /* Competitive analysis */
        .competitor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .competitor-card {
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Revenue model */
        .pricing-card {
            padding: 2rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .pricing-feature {
            padding: 0.5rem 0;
            border-bottom: 1px solid #93C5FD;
        }

        </style>
    """,
        unsafe_allow_html=True,
    )


def summary():
    st.header("How It Works")
    cols = st.columns(4)

    with cols[0]:
        st.markdown(
            """
            <br />

            <div class="feature-card">
                <h4>📤 Upload Data</h4>
                <p>Upload your investment data file or input manually</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            """
            <br />

            <div class="feature-card">
                <h4>📊 Explore Portfolio</h4>
                <p>View assets and performance metrics</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            """
            <br />

            <div class="feature-card">
                <h4>📈 Track Performance</h4>
                <p>Monitor growth and returns</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        st.markdown(
            """
            <br />

            <div class="feature-card">
                <h4>🎯 Optimize</h4>
                <p>Get insights and recommendations</p>
            </div>
        """,
            unsafe_allow_html=True,
        )


def go_to_market_strategy():
    st.header("Go-To-Market Strategy")

    st.markdown(
        """
        <div class="timeline">
            <div class="timeline-item">
                <strong>Months 1-3: Launch Preparation</strong>
                <ul>
                    <li>MVP development and testing</li>
                    <li>Market validation and surveys</li>
                    <li>Partnership building</li>
                </ul>
            </div>
            <div class="timeline-item">
                <strong>Months 4-7: Market Entry</strong>
                <ul>
                    <li>Product launch campaign</li>
                    <li>Initial user acquisition</li>
                    <li>Feedback collection and iteration</li>
                </ul>
            </div>
            <div class="timeline-item">
                <strong>Months 7-12: Growth Phase</strong>
                <ul>
                    <li>User retention programs</li>
                    <li>Product improvements</li>
                    <li>Community building</li>
                </ul>
            </div>
            <div class="timeline-item">
                <strong>Months 13-24: Scaling</strong>
                <ul>
                    <li>Market expansion</li>
                    <li>Advanced feature rollout</li>
                    <li>Monetization optimization</li>
                </ul>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def vision_mission_values():

    st.markdown(
        """
        <h5 style="font-size: 1.2rem; font-style: italic; color: white; text-align: center; margin-top: 1rem">
            "To simplify investment tracking with an intuitive platform that lets users monitor, analyze, and compare portfolio performance across asset classes."
        </h5>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="values-grid">
            <div class="value-card">
                <h4>🎯 Simplicity</h4>
                <p>Making complex finance simple</p>
            </div>
            <div class="value-card">
                <h4>🔍 Transparency</h4>
                <p>Clear and honest insights</p>
            </div>
            <div class="value-card">
                <h4>💪 Empowerment</h4>
                <p>Enabling informed decisions</p>
            </div>
            <div class="value-card">
                <h4>💡 Innovation</h4>
                <p>Cutting-edge solutions</p>
            </div>
            <div class="value-card">
                <h4>👥 Community</h4>
                <p>Growing together</p>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def product_overview():
    st.header("Product Overview")

    # Core Features
    st.subheader("Core Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="product-feature">
                <h4>📊 Portfolio Aggregation</h4>
                <p>Sync data from multiple sources including brokerage accounts, crypto wallets, and real estate</p>
            </div>
            <div class="product-feature">
                <h4>📈 Benchmark Comparison</h4>
                <p>Compare portfolios against market indexes or ETFs</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="product-feature">
                <h4>📱 Performance Dashboard</h4>
                <p>User-friendly display of key metrics (ROI, CAGR, diversification)</p>
            </div>
            <div class="product-feature">
                <h4>🎯 Goal Tracking</h4>
                <p>Set and monitor financial goals with progress tracking</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Differentiation
    st.subheader("Differentiation")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <h4>👥 Community-Driven</h4>
                <p>Compare performance with anonymized user data</p>
            </div>
            <div class="feature-card">
                <h4>📰 Sentiment Analysis</h4>
                <p>Track news, social media, and market trends</p>
            </div>
            <div class="feature-card">
                <h4>🤖 AI-Powered</h4>
                <p>Personalized portfolio optimization</p>
            </div>
            <div class="feature-card">
                <h4>⚡ Real-Time Alerts</h4>
                <p>Notifications for significant events</p>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def competitive_analysis():
    st.header("Competitive Analysis")

    st.markdown(
        """
        <div class="competitor-grid">
            <div class="competitor-card">
                <h4>Traditional Platforms</h4>
                <ul>
                    <li>Limited asset coverage</li>
                    <li>Complex interfaces</li>
                    <li>High costs</li>
                </ul>
            </div>
            <div class="competitor-card">
                <h4>Our Advantages</h4>
                <ul>
                    <li>Comprehensive asset tracking</li>
                    <li>AI-powered insights</li>
                    <li>Community features</li>
                    <li>Educational resources</li>
                </ul>
            </div>
            <div class="competitor-card">
                <h4>Market Gap</h4>
                <ul>
                    <li>User-friendly interface</li>
                    <li>Affordable pricing</li>
                    <li>Integrated learning tools</li>
                </ul>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def pricing_model():
    st.header("Pricing")

    st.markdown(
        """
        <div class="pricing-card">
            <h3>🆓 Free Tier</h3>
            <div class="pricing-feature">✓ Basic portfolio tracking</div>
            <div class="pricing-feature">✓ Performance overview</div>
            <div class="pricing-feature">✓ Limited benchmarking</div>
            <div class="pricing-feature">✓ Investment guides</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def conclusion():
    st.header("Conclusion")
    st.markdown(
        """
        <div style="text-align: center">
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                My Financial Vision is poised to become the go-to platform for investors seeking clarity and control over their portfolios. Through our intuitive, AI-driven, and community-supported investment tracking solution, we'll transform personal finance management in Italy and beyond.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":

    configure_page()

    st.session_state["loading_data"] = False
    load_upload_action_menu()

    st.markdown(
        """
        <h1 style="font-style: italic; text-align: center; margin-top: 1rem">
            My Financial Vision
        </h1>

        <h3 style="font-size: 1.2rem; font-style: italic; color: #2563EB; text-align: center; margin-top: 1rem">
            "Empowering individuals to make smarter financial decisions through clear, actionable insights into their investments."
        </h3>
    """,
        unsafe_allow_html=True,
    )
    summary()
    # vision_mission_values()
    product_overview()
    competitive_analysis()
    pricing_model()
    go_to_market_strategy()
    conclusion()
