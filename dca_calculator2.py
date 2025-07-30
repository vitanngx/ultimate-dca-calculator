import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from flask_caching import Cache

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Set up caching
cache = Cache(server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 200
})
cache.clear()

# Constants
DEFAULT_TICKER = 'VOO'
DEFAULT_INVESTMENT = 100
DEFAULT_FEE = 0
DEFAULT_START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
FREQUENCY_OPTIONS = [
    {'label': 'Daily', 'value': '1d'},
    {'label': 'Weekly', 'value': '1wk'},
    {'label': 'Monthly', 'value': '1mo'}
]

# Helper functions
@cache.memoize(timeout=7200)  # 2 hours cache
def fetch_stock_data(ticker, start_date, end_date, interval='1mo'):
    """Fetch stock data with caching and robust error handling"""
    try:
        # Validate ticker format
        ticker = ticker.upper().strip()
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty")
        
        print(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {interval}")
        
        # Try to download data
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker}. Please check the ticker symbol.")
        
        print(f"Raw data shape: {df.shape}")
        print(f"Raw data columns: {df.columns.tolist()}")
        print(f"Raw data index: {df.index.name}")
        
        # Handle multi-level columns (common with yfinance)
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            print("Multi-level columns detected, flattening...")
            # Flatten multi-level columns
            df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Handle different possible date column names from yfinance
        date_col = None
        possible_date_cols = ['Date', 'Datetime', 'date', 'datetime']
        
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # Sometimes the date is still in index
            if df.index.name in possible_date_cols:
                df = df.reset_index()
                date_col = df.columns[0]
        
        if date_col is None:
            available_cols = df.columns.tolist()
            raise ValueError(f"Could not find date column. Available columns: {available_cols}")
        
        # Rename to standard 'Date'
        if date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
            
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle price columns - be more flexible with column names
        price_col = None
        
        # First try standard price column names
        possible_price_cols = ['Adj Close', 'Close', 'close', 'adj_close', 'adjclose']
        
        for col in possible_price_cols:
            if col in df.columns:
                price_col = col
                break
        
        # If not found, check if ticker name appears as column (common with crypto)
        if price_col is None:
            ticker_cols = [col for col in df.columns if ticker in str(col).upper()]
            if ticker_cols:
                # Use the first ticker column as price data
                price_col = ticker_cols[0]
                print(f"Using ticker column '{price_col}' as price data")
                
                # If there are multiple identical columns, take just the first one
                if isinstance(df[price_col], pd.DataFrame):
                    df[price_col] = df[price_col].iloc[:, 0]  # Take first column if multiple
        
        # If still not found, try any numeric column that could be price
        if price_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove Date column if it's numeric
            numeric_cols = [col for col in numeric_cols if 'date' not in col.lower()]
            if numeric_cols:
                price_col = numeric_cols[0]  # Use first numeric column
                print(f"Using first numeric column '{price_col}' as price data")
        
        if price_col is None:
            available_cols = df.columns.tolist()
            raise ValueError(f"No price data found. Available columns: {available_cols}")
        
        # Rename to standard 'Adj Close' if needed
        if price_col != 'Adj Close':
            # Handle case where price_col might return multiple columns
            price_data = df[price_col]
            if isinstance(price_data, pd.DataFrame):
                # If multiple columns, take the first one
                df['Adj Close'] = price_data.iloc[:, 0]
            else:
                df['Adj Close'] = price_data
            print(f"Renamed '{price_col}' to 'Adj Close'")
        
        # Ensure we have valid price data
        if df['Adj Close'].isna().all():
            raise ValueError("All price data is missing (NaN values)")
        
        # Convert price to numeric if it's not already
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
        
        # Remove any rows with missing price data
        initial_len = len(df)
        df = df.dropna(subset=['Adj Close'])
        
        if len(df) == 0:
            raise ValueError("No valid price data after removing missing values")
            
        if len(df) != initial_len:
            print(f"Removed {initial_len - len(df)} rows with missing price data")
        
        # Debug info
        print(f"Final data for {ticker}: {len(df)} rows")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Final columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")
        
        return df
        
    except Exception as e:
        print(f"Error in fetch_stock_data: {str(e)}")
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

def get_current_price(ticker):
    """Get current/latest price for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try to get current price from info first
        info = stock.info
        current_price = None
        
        # Try different price fields
        price_fields = ['currentPrice', 'regularMarketPrice', 'price', 'ask', 'bid']
        for field in price_fields:
            if field in info and info[field] is not None:
                current_price = float(info[field])
                break
        
        # If info doesn't work, get latest price from recent data
        if current_price is None:
            recent_data = stock.history(period="1d", interval="1m")
            if not recent_data.empty:
                current_price = float(recent_data['Close'].iloc[-1])
        
        # Final fallback - use 5 day data
        if current_price is None:
            recent_data = stock.history(period="5d")
            if not recent_data.empty:
                current_price = float(recent_data['Close'].iloc[-1])
        
        return current_price
        
    except Exception as e:
        print(f"Error getting current price for {ticker}: {e}")
        return None

def calculate_dca(df, investment_amount, transaction_fee, ticker=None):
    """Calculate DCA results with vectorized operations"""
    df = df.copy()
    
    # Calculate shares purchased at each interval (after fees)
    net_investment = investment_amount - transaction_fee
    if net_investment <= 0:
        raise ValueError("Investment amount must be greater than transaction fee")
    
    df['Shares'] = net_investment / df['Adj Close']
    df['Invested'] = investment_amount  # Total amount including fees
    
    total_shares = df['Shares'].sum()
    total_invested = df['Invested'].sum()
    avg_price = total_invested / total_shares if total_shares > 0 else 0
    
    # Get current price (real-time) instead of end date price
    current_price = get_current_price(ticker) if ticker else None
    if current_price is None:
        # Fallback to last available price from data
        current_price = df['Adj Close'].iloc[-1]
        print(f"Using last available price: ${current_price:.2f}")
    else:
        print(f"Using current market price: ${current_price:.2f}")
    
    current_value = total_shares * current_price
    profit = current_value - total_invested
    profit_pct = (profit / total_invested) * 100 if total_invested > 0 else 0
    
    # Calculate cumulative values for chart
    df['Cumulative Shares'] = df['Shares'].cumsum()
    df['Cumulative Invested'] = df['Invested'].cumsum()
    df['Portfolio Value'] = df['Cumulative Shares'] * df['Adj Close']
    
    return {
        'total_invested': total_invested,
        'total_shares': total_shares,
        'avg_price': avg_price,
        'current_price': current_price,
        'current_value': current_value,
        'profit': profit,
        'profit_pct': profit_pct,
        'transactions': df[['Date', 'Adj Close', 'Shares', 'Invested']],
        'growth_data': df[['Date', 'Portfolio Value', 'Cumulative Invested']]
    }

def calculate_lump_sum(df, investment_amount, transaction_fee, ticker=None):
    """Calculate lump sum investment results"""
    initial_price = df['Adj Close'].iloc[0]
    net_investment = investment_amount - transaction_fee
    
    if net_investment <= 0:
        raise ValueError("Investment amount must be greater than transaction fee")
    
    shares_bought = net_investment / initial_price
    total_invested = investment_amount
    
    # Get current price (real-time) instead of end date price
    current_price = get_current_price(ticker) if ticker else None
    if current_price is None:
        # Fallback to last available price from data
        current_price = df['Adj Close'].iloc[-1]
        print(f"Using last available price: ${current_price:.2f}")
    else:
        print(f"Using current market price: ${current_price:.2f}")
    
    current_value = shares_bought * current_price
    profit = current_value - total_invested
    profit_pct = (profit / total_invested) * 100 if total_invested > 0 else 0
    
    # Prepare growth data for chart
    growth_data = df[['Date', 'Adj Close']].copy()
    growth_data['Portfolio Value'] = shares_bought * growth_data['Adj Close']
    growth_data['Cumulative Invested'] = total_invested
    
    return {
        'total_invested': total_invested,
        'total_shares': shares_bought,
        'avg_price': initial_price,
        'current_price': current_price,
        'current_value': current_value,
        'profit': profit,
        'profit_pct': profit_pct,
        'growth_data': growth_data
    }

# App layout
app.layout = dbc.Container([
    html.H1("DCA vs Lump Sum Investment Calculator", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Investment Parameters", className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Stock/ETF Ticker", html_for="ticker"),
                            dbc.Input(
                                id='ticker',
                                value=DEFAULT_TICKER,
                                type='text',
                                placeholder="e.g. VOO, AAPL, MSFT",
                                debounce=True
                            ),
                            dbc.FormText("Use Yahoo Finance ticker symbols"),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Investment Amount ($)", html_for="investment"),
                            dbc.Input(
                                id='investment',
                                value=DEFAULT_INVESTMENT,
                                type='number',
                                min=1,
                                step=1
                            ),
                            dbc.FormText("Amount to invest per period (for DCA) or total amount (for Lump Sum)"),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Transaction Fee ($)", html_for="fee"),
                            dbc.Input(
                                id='fee',
                                value=DEFAULT_FEE,
                                type='number',
                                min=0,
                                step=0.01
                            ),
                            dbc.FormText("Fee per transaction (applied to each DCA purchase or once for Lump Sum)"),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Frequency", html_for="frequency"),
                            dcc.Dropdown(
                                id='frequency',
                                options=FREQUENCY_OPTIONS,
                                value='1mo',
                                clearable=False
                            ),
                            dbc.FormText("How often to invest (DCA only)"),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Date", html_for="start-date"),
                            dcc.DatePickerSingle(
                                id='start-date',
                                date=DEFAULT_START_DATE,
                                max_date_allowed=DEFAULT_END_DATE,
                                display_format='YYYY-MM-DD'
                            ),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("End Date", html_for="end-date"),
                            dcc.DatePickerSingle(
                                id='end-date',
                                date=DEFAULT_END_DATE,
                                max_date_allowed=DEFAULT_END_DATE,
                                display_format='YYYY-MM-DD'
                            ),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Investment Type", html_for="investment-type"),
                            dcc.Dropdown(
                                id='investment-type',
                                options=[
                                    {'label': 'Dollar Cost Averaging (DCA)', 'value': 'dca'},
                                    {'label': 'Lump Sum Investment', 'value': 'lump'},
                                    {'label': 'Compare Both', 'value': 'both'}
                                ],
                                value='both',
                                clearable=False
                            ),
                        ], className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Calculate", id='calculate-btn', color="primary", className="w-100"),
                        ]),
                    ]),
                ])
            ]),
            
            dbc.Card([
                dbc.CardHeader("Information", className="bg-info text-white"),
                dbc.CardBody([
                    html.P("This calculator compares Dollar Cost Averaging (DCA) with Lump Sum investing."),
                    html.P("DCA invests a fixed amount at regular intervals, while Lump Sum invests everything upfront."),
                    html.P("When comparing both strategies, Lump Sum uses the same total amount as DCA would invest over the entire period."),
                    html.P("💡 Current value calculations use real-time market prices when available.", className="text-success"),
                    html.P("Data is fetched from Yahoo Finance.", className="mb-0"),
                ])
            ], className="mt-3"),
        ], width=4),
        
        dbc.Col([
            dcc.Loading(
                id="loading-main",
                type="default",
                children=[
                    html.Div(id='results-container'),
                ]
            ),
            
            dbc.Card([
                dbc.CardHeader("Investment Growth", className="bg-primary text-white"),
                dbc.CardBody([
                    dcc.Graph(id='growth-chart', config={'displayModeBar': False}),
                ])
            ], className="mt-3"),
            
            dbc.Card([
                dbc.CardHeader("Transactions", className="bg-secondary text-white"),
                dbc.CardBody([
                    html.Div(id='transactions-table'),
                ])
            ], className="mt-3"),
        ], width=8)
    ]),
    
    # Hidden div to store raw data
    html.Div(id='raw-data', style={'display': 'none'}),
    
    # Error modal
    dbc.Modal([
        dbc.ModalHeader("Error"),
        dbc.ModalBody(id='error-message'),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-error-modal", className="ml-auto")
        ),
    ], id="error-modal"),
], fluid=True)

# Callbacks
@app.callback(
    [Output('raw-data', 'children'),
     Output('error-modal', 'is_open'),
     Output('error-message', 'children')],
    [Input('calculate-btn', 'n_clicks'),
     Input('close-error-modal', 'n_clicks')],
    [State('ticker', 'value'),
     State('investment', 'value'),
     State('fee', 'value'),
     State('frequency', 'value'),
     State('start-date', 'date'),
     State('end-date', 'date'),
     State('error-modal', 'is_open')],
    prevent_initial_call=True
)
def handle_modal_and_data(calc_clicks, close_clicks, ticker, investment, fee, frequency, start_date, end_date, is_open):
    """Handle both data fetching and modal closing"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle close modal button
    if trigger_id == 'close-error-modal':
        return dash.no_update, False, dash.no_update
    
    # Handle calculate button
    if trigger_id == 'calculate-btn':
        # Input validation
        if not ticker or not ticker.strip():
            return None, True, "Please enter a valid ticker symbol"
        
        if not investment or investment <= 0:
            return None, True, "Investment amount must be greater than 0"
        
        if fee < 0:
            return None, True, "Transaction fee cannot be negative"
        
        if fee >= investment:
            return None, True, "Transaction fee cannot be greater than or equal to investment amount"
        
        # Date validation
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt >= end_dt:
                return None, True, "Start date must be before end date"
            
            if end_dt > pd.to_datetime('today'):
                return None, True, "End date cannot be in the future"
                
            # Check if date range is too short
            if (end_dt - start_dt).days < 30:
                return None, True, "Date range must be at least 30 days"
                
        except:
            return None, True, "Invalid date format"
        
        try:
            df = fetch_stock_data(ticker.strip(), start_date, end_date, frequency)
            
            # Additional validation - ensure we have enough data points
            if len(df) < 2:
                return None, True, f"Not enough data points for {ticker}. Try a longer date range or different frequency."
                
            return df.to_json(date_format='iso', orient='split'), False, ""
        except Exception as e:
            return None, True, str(e)
    
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('results-container', 'children'),
     Output('growth-chart', 'figure'),
     Output('transactions-table', 'children')],
    [Input('raw-data', 'children'),
     Input('investment-type', 'value')],
    [State('investment', 'value'),
     State('fee', 'value'),
     State('ticker', 'value')]
)
def update_output(json_data, investment_type, investment, fee, ticker):
    """Update all output components with improved comparison logic"""
    if not json_data:
        return [
            dbc.Card([
                dbc.CardHeader("Results", className="bg-success text-white"),
                dbc.CardBody([
                    html.P("Enter parameters and click Calculate to see results", className="text-center")
                ])
            ])
        ], go.Figure(), None
    
    try:
        df = pd.read_json(json_data, orient='split')
        
        # Debug: print column names to see what we have
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        
        # Handle different possible date column names
        date_col = None
        possible_date_cols = ['Date', 'date', 'Datetime', 'datetime', 'index']
        
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # If no date column found, reset index might help
            df = df.reset_index()
            if 'Date' in df.columns:
                date_col = 'Date'
            elif 'index' in df.columns:
                date_col = 'index'
                df = df.rename(columns={'index': 'Date'})
                date_col = 'Date'
        
        if date_col and date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
        
        # Ensure Date column exists
        if 'Date' not in df.columns:
            raise ValueError("No date column found in the data")
            
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check for Adj Close column
        if 'Adj Close' not in df.columns:
            available_cols = df.columns.tolist()
            raise ValueError(f"No 'Adj Close' column found. Available columns: {available_cols}")
        
        # Validate data
        if df.empty:
            raise ValueError("DataFrame is empty after processing")
            
        if df['Adj Close'].isna().all():
            raise ValueError("All price data is missing")
        
        # Remove rows with missing price data
        initial_len = len(df)
        df = df.dropna(subset=['Adj Close'])
        if len(df) == 0:
            raise ValueError("No valid price data available")
        
        if len(df) != initial_len:
            print(f"Removed {initial_len - len(df)} rows with missing price data")
        
        results = []
        fig = go.Figure()
        transactions_table = None
        
        dca_results = None
        lump_results = None
        
        if investment_type in ['dca', 'both']:
            dca_results = calculate_dca(df, investment, fee, ticker)
            results.append(dbc.Card([
                dbc.CardHeader("DCA Results", className="bg-info text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Invested", className="card-title"),
                            html.H4(f"${dca_results['total_invested']:,.2f}", className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Current Value", className="card-title"),
                            html.H4(f"${dca_results['current_value']:,.2f}", className="card-text"),
                        ]),
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Total Shares: {dca_results['total_shares']:,.4f}"),
                            html.P(f"Average Price: ${dca_results['avg_price']:,.2f}"),
                            html.P(f"Number of Purchases: {len(df)}"),
                        ]),
                        dbc.Col([
                            html.P(f"Current Price: ${dca_results['current_price']:,.2f}"),
                            html.P([
                                "Total Profit: ",
                                html.Span(
                                    f"${dca_results['profit']:,.2f} ({dca_results['profit_pct']:,.2f}%)",
                                    className="text-success" if dca_results['profit'] >= 0 else "text-danger"
                                )
                            ]),
                            html.P(f"Total Fees Paid: ${fee * len(df):,.2f}"),
                        ]),
                    ]),
                ])
            ], className="mb-3"))
            
            fig.add_trace(go.Scatter(
                x=dca_results['growth_data']['Date'],
                y=dca_results['growth_data']['Portfolio Value'],
                name='DCA Portfolio Value',
                line=dict(color='blue', width=2),
                hovertemplate='<b>DCA Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=dca_results['growth_data']['Date'],
                y=dca_results['growth_data']['Cumulative Invested'],
                name='DCA Amount Invested',
                line=dict(color='blue', dash='dot', width=1),
                hovertemplate='<b>DCA Invested</b><br>Date: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
            ))
            
            if investment_type == 'dca':
                transactions_df = dca_results['transactions'].copy()
                transactions_df['Date'] = transactions_df['Date'].dt.strftime('%Y-%m-%d')
                transactions_df['Adj Close'] = transactions_df['Adj Close'].round(2)
                transactions_df['Shares'] = transactions_df['Shares'].round(4)
                transactions_df['Invested'] = transactions_df['Invested'].round(2)
                transactions_df.columns = ['Date', 'Price', 'Shares Bought', 'Amount Invested']
                
                transactions_table = dbc.Table.from_dataframe(
                    transactions_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm"
                )
        
        if investment_type in ['lump', 'both']:
            # For comparison, lump sum should invest the same total amount as DCA
            if investment_type == 'both':
                lump_sum_amount = investment * len(df)  # Total that would be invested via DCA
            else:
                lump_sum_amount = investment
                
            lump_results = calculate_lump_sum(df, lump_sum_amount, fee, ticker)
            
            results.append(dbc.Card([
                dbc.CardHeader("Lump Sum Results", className="bg-warning text-dark"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Invested", className="card-title"),
                            html.H4(f"${lump_results['total_invested']:,.2f}", className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Current Value", className="card-title"),
                            html.H4(f"${lump_results['current_value']:,.2f}", className="card-text"),
                        ]),
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Total Shares: {lump_results['total_shares']:,.4f}"),
                            html.P(f"Purchase Price: ${lump_results['avg_price']:,.2f}"),
                            html.P("Number of Purchases: 1"),
                        ]),
                        dbc.Col([
                            html.P(f"Current Price: ${lump_results['current_price']:,.2f}"),
                            html.P([
                                "Total Profit: ",
                                html.Span(
                                    f"${lump_results['profit']:,.2f} ({lump_results['profit_pct']:,.2f}%)",
                                    className="text-success" if lump_results['profit'] >= 0 else "text-danger"
                                )
                            ]),
                            html.P(f"Total Fees Paid: ${fee:,.2f}"),
                        ]),
                    ]),
                ])
            ], className="mb-3"))
            
            fig.add_trace(go.Scatter(
                x=lump_results['growth_data']['Date'],
                y=lump_results['growth_data']['Portfolio Value'],
                name='Lump Sum Portfolio Value',
                line=dict(color='orange', width=2),
                hovertemplate='<b>Lump Sum Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=lump_results['growth_data']['Date'],
                y=lump_results['growth_data']['Cumulative Invested'],
                name='Lump Sum Amount Invested',
                line=dict(color='orange', dash='dot', width=1),
                hovertemplate='<b>Lump Sum Invested</b><br>Date: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
            ))
        
        # Add comparison summary when both strategies are shown
        if investment_type == 'both' and dca_results and lump_results:
            difference = dca_results['current_value'] - lump_results['current_value']
            difference_pct = (difference / lump_results['current_value']) * 100
            better_strategy = "DCA" if difference > 0 else "Lump Sum"
            better_color = "success" if difference > 0 else "danger"
            
            results.append(dbc.Card([
                dbc.CardHeader("Strategy Comparison", className="bg-dark text-white"),
                dbc.CardBody([
                    html.H5(f"{better_strategy} performed better", className=f"text-{better_color}"),
                    html.P([
                        "Difference: ",
                        html.Span(
                            f"${abs(difference):,.2f} ({abs(difference_pct):,.2f}%)",
                            className=f"text-{better_color}"
                        )
                    ]),
                    html.Hr(),
                    html.H6("Summary:"),
                    html.P(f"DCA: ${dca_results['current_value']:,.2f} ({dca_results['profit_pct']:,.2f}% return)"),
                    html.P(f"Lump Sum: ${lump_results['current_value']:,.2f} ({lump_results['profit_pct']:,.2f}% return)"),
                ])
            ], className="mb-3"))
        
        # Update chart layout
        fig.update_layout(
            title=f"{ticker.upper()} Investment Growth Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1
            ),
            yaxis=dict(tickformat="$,.0f")
        )
        
        return results, fig, transactions_table
    
    except Exception as e:
        error_card = dbc.Card([
            dbc.CardHeader("Error", className="bg-danger text-white"),
            dbc.CardBody([
                html.P(f"Error processing data: {str(e)}", className="text-danger")
            ])
        ])
        return [error_card], go.Figure(), None

# Run the app
if __name__ == '__main__':
    app.run(debug=True)