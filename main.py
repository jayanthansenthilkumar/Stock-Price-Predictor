import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

def get_stock_data(ticker, start_date, end_date):
    """Download stock data safely with column validation."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Verify required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing required columns in data for {ticker}")
            return None
            
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
            
        # Handle any NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_prediction(prices):
    """Calculate linear regression prediction with error handling."""
    try:
        if len(prices) < 2:
            st.error("Insufficient data for prediction")
            return None, None, None
            
        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        
        # Calculate coefficients
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope (b1)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if abs(denominator) < 1e-10:  # Avoid division by zero
            st.warning("Unable to calculate prediction due to constant prices")
            return None, None, None
            
        slope = numerator / denominator
        
        # Calculate intercept (b0)
        intercept = y_mean - slope * x_mean
        
        # Make prediction for next day
        next_day = len(prices)
        prediction = float(slope * next_day + intercept)
        
        return prediction, float(slope), float(intercept)
        
    except Exception as e:
        st.error(f"Error in prediction calculation: {str(e)}")
        return None, None, None

def plot_stock_data(data, predictions=None, title="Stock Price"):
    """Create stock price plot using Plotly with error handling."""
    try:
        fig = go.Figure()
        
        # Verify data is valid
        if data is None or data.empty:
            st.error("No data available for plotting")
            return None
            
        # Plot actual prices
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Actual Price',
            line=dict(color='blue')
        ))
        
        # Plot predictions if available
        if predictions is not None and not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions.index,
                y=predictions,
                name='Predicted Price',
                line=dict(color='orange', dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    
    st.title("📈 Stock Price Predictor")
    st.write("A simple stock price prediction tool using linear regression")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Parameters")
        
        # Stock symbol input with validation
        symbol = st.text_input("Stock Symbol", "AAPL")
        symbol = symbol.upper().strip()
        
        if not symbol:
            st.warning("Please enter a stock symbol")
            return
        
        # Date range selection
        today = date.today()
        default_start = today - timedelta(days=365)
        
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=today
        )
        
        end_date = st.date_input(
            "End Date",
            value=today,
            max_value=today
        )
        
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

    # Load data with progress indicator
    with st.spinner("Fetching stock data..."):
        df = get_stock_data(symbol, start_date, end_date)
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
        
    try:
        # Display basic stock info
        st.subheader(f"{symbol} Stock Analysis")
        
        # Safely get latest prices
        current_price = float(df['Close'].iloc[-1])
        previous_price = float(df['Close'].iloc[-2])
        
        # Calculate metrics
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Price Change", f"${price_change:.2f}", 
                     f"{price_change_pct:+.2f}%")
        with col3:
            st.metric("Trading Days", len(df))
        
        # Calculate predictions
        prices = df['Close'].values
        next_price, slope, intercept = calculate_prediction(prices)
        
        if next_price is not None:
            # Generate prediction line
            x_pred = np.arange(len(prices) + 1)
            y_pred = slope * x_pred + intercept
            pred_series = pd.Series(
                y_pred, 
                index=list(df.index) + [df.index[-1] + pd.Timedelta(days=1)]
            )
            
            # Create and display plot
            fig = plot_stock_data(df, pred_series, f"{symbol} Stock Price and Prediction")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction
            pred_change = next_price - current_price
            pred_change_pct = (pred_change / current_price) * 100
            
            st.subheader("Price Prediction")
            st.metric(
                "Next Day Prediction",
                f"${next_price:.2f}",
                f"{pred_change_pct:+.2f}%"
            )
            
            # Additional statistics with error handling
            # st.subheader("Summary Statistics")
            # stats_col1, stats_col2 = st.columns(2)
            
            # with stats_col1:
            #     st.write("Historical Stats:")
            #     st.write(f"Highest Price: ${df['High'].max():.2f}")
            #     st.write(f"Lowest Price: ${df['Low'].min():.2f}")
            #     st.write(f"Average Volume: {int(df['Volume'].mean()):,}")
                
            # with stats_col2:
            #     st.write("Trend Analysis:")
            #     ma30 = df['Close'].rolling(30).mean().iloc[-1]
            #     ma7 = df['Close'].rolling(7).mean().iloc[-1]
            #     st.write(f"30-Day Average: ${ma30:.2f}")
            #     st.write(f"7-Day Average: ${ma7:.2f}")
        
        # Show raw data if requested
        if st.checkbox("Show Raw Data"):
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.write("Please try again with different parameters or contact support if the issue persists.")

if __name__ == "__main__":
    main()