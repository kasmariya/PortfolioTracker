from flask import Flask, render_template, url_for
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os

app = Flask(__name__)

portfolio = {
    'TATAMOTORS.NS': (600, 740.42),
    'JIOFIN.NS': (1550, 228.49),
    'TCS.NS': (79, 3472.58),
    'TATAPOWER.NS': (550, 318.05),
    'BEL.NS': (580, 249.54),
    'IRCTC.NS': (225, 715.00),
    'TITAN.NS': (50, 3016.05),
    'MOTHERSON.NS': (1000, 124.14),
    'HINDUNILVR.NS': (50, 2178.46),
    'BAJAJHFL.NS': (690, 114.77)
}

CACHE_FILE = 'financials_cache.pkl'
financials_cache = {}

# Load/Save Cache
def load_cache():
    global financials_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            financials_cache = pickle.load(f)

def save_cache():
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(financials_cache, f)

# Manual ROE Calculation
def calculate_manual_roe(ticker):
    try:
        if ticker not in financials_cache:
            stock = yf.Ticker(ticker)
            financials_cache[ticker] = {
                'income_stmt': stock.financials,
                'balance_sheet': stock.balance_sheet
            }
            save_cache()

        income_stmt = financials_cache[ticker]['income_stmt']
        balance_sheet = financials_cache[ticker]['balance_sheet']

        net_income = None
        shareholder_equity = None

        for ni in ['Net Income', 'NetIncome', 'Net Income Common Stockholders']:
            if ni in income_stmt.index:
                net_income = income_stmt.loc[ni].iloc[0]
                break

        for eq in ['Total Stockholder Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']:
            if eq in balance_sheet.index:
                shareholder_equity = balance_sheet.loc[eq].iloc[0]
                break

        if net_income is None or shareholder_equity is None or shareholder_equity == 0:
            return None

        roe = (net_income / shareholder_equity) * 100
        return round(roe, 2)

    except Exception as e:
        print(f"Error calculating manual ROE for {ticker}: {e}")
        return None

# Core Functions
def get_portfolio_data():
    ticker_values = {}
    sector_values = defaultdict(float)
    rows = []
    total_cost = 0.0
    total_value = 0.0

    for ticker, (qty, buy_price) in portfolio.items():
