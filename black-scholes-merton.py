import yfinance as yf
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from fredapi import Fred
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import time
import shutil


# FRED API setup
os.environ['FRED_API_KEY'] = 'PUT YOUR API KEY HERE'
fred = Fred(api_key=os.environ['FRED_API_KEY'])
CACHE_FILE = 'treasury_yields_cache.json'

def fetch_and_cache_treasury_yields():
    print("Fetching and caching Treasury yields...")
    series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
    data = {}
    for series_id in series_ids:
        yield_data = fred.get_series(series_id)
        data[series_id] = yield_data.iloc[-1] / 100  # Convert percentage to decimal
    data['timestamp'] = datetime.now().isoformat()
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f)
    return data

def get_cached_treasury_yields():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        last_update = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - last_update < timedelta(days=1):
            return data
    return fetch_and_cache_treasury_yields()

def get_risk_free_rate(days_to_expiration):
    yields = get_cached_treasury_yields()
    if days_to_expiration <= 30:
        return yields['DGS1MO']
    elif days_to_expiration <= 90:
        return yields['DGS3MO']
    elif days_to_expiration <= 180:
        return yields['DGS6MO']
    elif days_to_expiration <= 365:
        return yields['DGS1']
    elif days_to_expiration <= 730:
        return yields['DGS2']
    elif days_to_expiration <= 1825:
        return yields['DGS5']
    elif days_to_expiration <= 3650:
        return yields['DGS10']
    else:
        return yields['DGS30']

@lru_cache(maxsize=None)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].tolist()

def print_aligned(ticker, message):
    console_width = shutil.get_terminal_size().columns
    ticker_width = 6  # Adjust if needed for longer tickers
    message_width = len(message)
    spacing = console_width - ticker_width - message_width - 1  # -1 for safety
    print(f"{ticker:<{ticker_width}}{' ' * spacing}{message}")

def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    
    if 'regularMarketPrice' in stock.info and stock.info['regularMarketPrice'] is not None:
        return stock.info['regularMarketPrice']
    elif 'currentPrice' in stock.info and stock.info['currentPrice'] is not None:
        return stock.info['currentPrice']
    elif 'previousClose' in stock.info and stock.info['previousClose'] is not None:
        return stock.info['previousClose']
    else:
        hist = stock.history(period="1d")
        if not hist.empty and 'Close' in hist.columns:
            return hist['Close'].iloc[-1]
        else:
            raise ValueError(f"Unable to fetch current price for {ticker}")

@torch.jit.script
def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.))))

@torch.jit.script
def black_scholes_merton(S, K, T, r, q, sigma, option_type):
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    
    call_price = S * torch.exp(-q * T) * norm_cdf(d1) - K * torch.exp(-r * T) * norm_cdf(d2)
    put_price = K * torch.exp(-r * T) * norm_cdf(-d2) - S * torch.exp(-q * T) * norm_cdf(-d1)
    
    return torch.where(option_type == 0, call_price, put_price)

def calculate_greeks(S, K, T, r, q, sigma, option_type):
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    delta = torch.exp(-q * T) * norm_cdf(d1) if option_type == 0 else -torch.exp(-q * T) * norm_cdf(-d1)
    gamma = torch.exp(-q * T) * norm_cdf(d1) / (S * sigma * torch.sqrt(T))
    theta = -((S * sigma * torch.exp(-q * T)) / (2 * torch.sqrt(T))) * norm_cdf(d1) - \
            r * K * torch.exp(-r * T) * norm_cdf(d2) + q * S * torch.exp(-q * T) * norm_cdf(d1)
    vega = S * torch.exp(-q * T) * torch.sqrt(T) * norm_cdf(d1)
    rho = K * T * torch.exp(-r * T) * norm_cdf(d2) if option_type == 0 else -K * T * torch.exp(-r * T) * norm_cdf(-d2)

    return delta, gamma, theta, vega, rho

def american_option_approximation(S, K, T, r, q, sigma, option_type):
    european_price = black_scholes_merton(S, K, T, r, q, sigma, option_type)
    
    if option_type == 0:  # Call option
        return torch.max(european_price, S - K)
    else:  # Put option
        return torch.max(european_price, K - S)

def price_options(options, stock_price):
    S = torch.tensor(stock_price).float()
    K = torch.tensor(options['strike'].values).float()
    T = torch.tensor(options['daysToExpiration'].values / 365).float()
    r = torch.tensor(options['riskFreeRate'].values).float()
    q = torch.tensor(options['dividendYield'].values).float()
    sigma = torch.tensor(options['impliedVolatility'].values).float()
    option_type = torch.tensor(options['optionType'].map({'call': 0, 'put': 1}).values).float()

    prices = []
    for i in range(len(K)):
        price = american_option_approximation(S, K[i], T[i], r[i], q[i], sigma[i], option_type[i])
        prices.append(price.item())
    
    return np.array(prices)

def get_user_input():
    print("Select the analysis option:")
    print("1. S&P 500")
    print("2. Specific Stock")
    
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if choice == '2':
        ticker = input("Enter the stock ticker symbol: ").upper()
    else:
        ticker = None

    print("\nSelect the price type to use:")
    print("1. Mark Price (midpoint between bid and ask)")
    print("2. Ask Price (typically used for buying an option)")
    print("3. Bid Price (typically used for selling an option)")
    
    while True:
        price_type_choice = input("Enter your choice (1, 2, or 3): ")
        if price_type_choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    price_type = ['mark', 'ask', 'bid'][int(price_type_choice) - 1]

    print("\nSelect the option type to analyze:")
    print("1. Calls")
    print("2. Puts")
    print("3. Both")
    
    while True:
        option_type_choice = input("Enter your choice (1, 2, or 3): ")
        if option_type_choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    option_type = ['call', 'put', 'both'][int(option_type_choice) - 1]

    print("\nEnter the following parameters (type 'NONE' if you don't want to set a limit):")
    
    min_price = input(f"Minimum {price_type} price: ")
    min_price = float(min_price) if min_price.lower() != 'none' else None
    
    max_price = input(f"Maximum {price_type} price: ")
    max_price = float(max_price) if max_price.lower() != 'none' else None
    
    min_open_interest = input("Minimum open interest: ")
    min_open_interest = int(min_open_interest) if min_open_interest.lower() != 'none' else None
    
    return choice, ticker, price_type, option_type, min_price, max_price, min_open_interest


def get_option_data(ticker, price_type, option_type, min_price, max_price, min_open_interest):
    stock = yf.Ticker(ticker)
    options = pd.DataFrame()
    
    try:
        expirations = stock.options
        if not expirations:
            print_aligned(ticker, "No options available")
            return options

        one_year_from_now = datetime.now() + timedelta(days=365)

        for expiration in expirations:
            expiration_date = datetime.strptime(expiration, '%Y-%m-%d')
            if expiration_date > one_year_from_now:
                break  # Stop processing expirations more than 1 year away

            try:
                calls = stock.option_chain(expiration).calls
                puts = stock.option_chain(expiration).puts
                calls['optionType'] = 'call'
                puts['optionType'] = 'put'
                calls['expiration'] = expiration
                puts['expiration'] = expiration
                
                if option_type == 'call':
                    options = pd.concat([options, calls])
                elif option_type == 'put':
                    options = pd.concat([options, puts])
                else:  # both
                    options = pd.concat([options, calls, puts])
            except Exception as e:
                print_aligned(ticker, f"Error fetching for options expiring on {expiration}: {e}")

        if options.empty:
            print_aligned(ticker, "No valid options data found")
            return options

        options['expirationDate'] = pd.to_datetime(options['expiration'])
        options['daysToExpiration'] = (options['expirationDate'] - datetime.now()).dt.days
        options['riskFreeRate'] = options['daysToExpiration'].apply(get_risk_free_rate)
        options['markPrice'] = (options['bid'] + options['ask']) / 2
        options['dividendYield'] = stock.info.get('dividendYield', 0)
        
        # Use the selected price type
        if price_type == 'mark':
            options['selectedPrice'] = options['markPrice']
        elif price_type == 'ask':
            options['selectedPrice'] = options['ask']
        else:  # bid
            options['selectedPrice'] = options['bid']
        
        # Early filtering
        options = options[
            (options['selectedPrice'] >= min_price if min_price is not None else True) &
            (options['selectedPrice'] <= max_price if max_price is not None else True) &
            (options['openInterest'] >= min_open_interest if min_open_interest is not None else True) &
            (options['expirationDate'] <= one_year_from_now)
        ]

    except Exception as e:
        print_aligned(ticker, f"Error fetching options data: {e}")
        return pd.DataFrame()

    return options

def analyze_options(ticker, price_type, option_type, min_price, max_price, min_open_interest):
    try:
        current_price = get_current_price(ticker)
        options = get_option_data(ticker, price_type, option_type, min_price, max_price, min_open_interest)
        
        if options.empty:
            print_aligned(ticker, "No options meeting the criteria")
            return pd.DataFrame()

        theoretical_prices = price_options(options, current_price)
        options['theoreticalPrice'] = theoretical_prices
        options['marketPrice'] = options['selectedPrice']
        options['priceDifference'] = options['theoreticalPrice'] - options['marketPrice']
        options['percentDifference'] = options['priceDifference'] / options['marketPrice'] * 100

        underpriced_options = options[options['percentDifference'] > 15].sort_values('percentDifference', ascending=False)
        return underpriced_options
    except Exception as e:
        print(f"{ticker} - Error analyzing: {e}")
        return pd.DataFrame()

def process_stock(ticker, price_type, option_type, min_price, max_price, min_open_interest):
    try:
        underpriced = analyze_options(ticker, price_type, option_type, min_price, max_price, min_open_interest)
        if not underpriced.empty:
            print_aligned(ticker, "Underpriced options found")
            underpriced['ticker'] = ticker
            return underpriced
        else:
            print_aligned(ticker, "No options meeting the criteria")
            return pd.DataFrame()
    except Exception as e:
        print_aligned(ticker, f"Error processing: {e}")
        return pd.DataFrame()

def main():
    start_time = time.time()  # Record the start time

    print("Stock Option Analyzer")
    choice, specific_ticker, price_type, option_type, min_price, max_price, min_open_interest = get_user_input()
    
    if choice == '1':
        print("Analyzing S&P 500 stocks...")
        tickers = get_sp500_tickers()
        index_name = "S&P 500"
    else:
        print(f"Analyzing specific stock: {specific_ticker}")
        tickers = [specific_ticker]
        index_name = f"Specific Stock ({specific_ticker})"
    
    all_underpriced_options = pd.DataFrame()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_stock, ticker, price_type, option_type, min_price, max_price, min_open_interest) for ticker in tickers]
        for future in as_completed(futures):
            result = future.result()
            if not result.empty:
                all_underpriced_options = pd.concat([all_underpriced_options, result])

    if not all_underpriced_options.empty:
        all_underpriced_options = all_underpriced_options.sort_values(['expirationDate', 'percentDifference'], ascending=[True, False])
        
        with open('output.txt', 'w') as f:
            f.write(f"Underpriced Options for {index_name} (Sorted by Date, then Percent Difference)\n")
            f.write("======================================================\n\n")
            f.write(f"Filters applied:\n")
            f.write(f"Analysis: {index_name}\n")
            f.write(f"Price type used: {price_type.capitalize()} price\n")
            f.write(f"Option type analyzed: {option_type.capitalize()}\n")
            f.write(f"Minimum {price_type} price: {min_price if min_price is not None else 'None'}\n")
            f.write(f"Maximum {price_type} price: {max_price if max_price is not None else 'None'}\n")
            f.write(f"Minimum open interest: {min_open_interest if min_open_interest is not None else 'None'}\n")
            f.write(f"Expiration date limit: 1 year from today ({datetime.now().date() + timedelta(days=365)})\n\n")
            
            for _, option in all_underpriced_options.iterrows():
                f.write(f"Ticker: {option['ticker']}\n")
                f.write(f"Expiration Date: {option['expirationDate']}\n")
                f.write(f"Option Type: {option['optionType']}\n")
                f.write(f"Strike: ${option['strike']:.2f}\n")
                f.write(f"{price_type.capitalize()} Price: ${option['marketPrice']:.2f}\n")
                f.write(f"Theoretical Price: ${option['theoreticalPrice']:.2f}\n")
                f.write(f"Percent Difference: {option['percentDifference']:.2f}%\n")
                f.write(f"Open Interest: {option['openInterest']}\n")
                f.write(f"Implied Volatility: {option['impliedVolatility']:.4f}\n")
                f.write(f"Days to Expiration: {option['daysToExpiration']}\n")
                f.write("\n")

        print(f"\nResults have been written to 'output.txt'")
    else:
        print(f"\nNo options meeting the criteria were found for the selected analysis ({index_name}).")

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the total runtime

    print(f"\nTotal runtime: {runtime:.2f} seconds")

if __name__ == "__main__":
    main()