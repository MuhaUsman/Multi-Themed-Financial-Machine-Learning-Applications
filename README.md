# E-Corp StockVerse ⚡

**Market Simulation Protocol Initialized**

E-Corp StockVerse is a comprehensive Streamlit web application designed for advanced stock market analysis, visualization, and AI-powered prediction. It features a dynamic, multi-themed interface inspired by popular series like Mr. Robot, Alice in Borderland, Squid Game, and Money Heist.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL_HERE)  <!-- Replace YOUR_STREAMLIT_APP_URL_HERE with your actual app URL after deployment -->

## Features

*   **Themed User Interface:**
    *   **Welcome Page (E-Corp/Dark Army):** A glitchy, terminal-style interface to set your target stock or upload CSV data.
    *   **Stock Arena (Alice in Borderland):** In-depth stock analysis with price charts, volume, RSI, returns distribution, rolling volatility, correlation heatmaps, seasonal decomposition, autocorrelation, and stationarity tests.
    *   **Squid ML (Squid Game):** Machine learning challenges for predicting stock prices (Linear Regression) and trends (Logistic Regression), plus K-means clustering analysis. Includes a visual ML pipeline.
    *   **Intelligence Hub (Money Heist):** A dashboard summarizing key metrics, technical indicators (MA Crossover, Bollinger Bands, RSI Trend, Volume Trend, MACD), and statistical insights.
*   **Data Sources:**
    *   Fetch live stock data using Yahoo Finance (yfinance).
    *   Upload custom stock data via CSV files with intelligent column mapping.
*   **Advanced Analytics:**
    *   Moving Averages (MA20, MA50)
    *   Relative Strength Index (RSI)
    *   Price Prediction (Linear Regression) with Confidence Intervals
    *   Trend Prediction (Logistic Regression) with Probability and Confusion Matrix
    *   K-means Clustering for pattern identification
    *   Seasonal Decomposition
    *   Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots
    *   Augmented Dickey-Fuller (ADF) Test for stationarity
    *   Feature Correlation Heatmaps
*   **Interactive Visualizations:**
    *   Candlestick charts, volume bars, RSI plots.
    *   Histograms for returns distribution.
    *   Area charts for rolling volatility.
    *   Gauge charts for trend confidence.
*   **Caching:** Utilizes Streamlit's caching for faster data loading on subsequent visits.

## Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** Web application framework.
*   **yfinance:** For fetching stock data from Yahoo Finance.
*   **Pandas:** Data manipulation and analysis.
*   **NumPy:** Numerical computations.
*   **Plotly:** Interactive charting and visualizations.
*   **Scikit-learn:** Machine learning models and metrics.
*   **Statsmodels:** Statistical models and time series analysis.
*   **Pillow (PIL):** Basic image operations (though mostly for SVGs/Base64 in this app).
*   **HTML/CSS:** For custom styling and theming.

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ecorp-stockverse-app.git
    cd ecorp-stockverse-app
    ```
    *(Replace `your-username/ecorp-stockverse-app.git` with your actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application should now be running and accessible in your web browser, typically at `http://localhost:8501`.

## Deployment

This application is designed to be deployed on Streamlit Community Cloud.

1.  Ensure your `app.py` and `requirements.txt` files are pushed to a public GitHub repository.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Click "New app" and connect your GitHub account.
4.  Select the repository, branch, and `app.py` as the main file.
5.  Deploy!

## File Structure
