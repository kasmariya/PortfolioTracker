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
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            info = stock.info

            if not hist.empty and len(hist) >= 1:
                current_price = hist['Close'].iloc[-1]
                market_value = current_price * qty
                cost_value = buy_price * qty
                gain_loss = market_value - cost_value

                total_cost += cost_value
                total_value += market_value

                sector = info.get("sector", "Unknown")
                ticker_values[ticker] = market_value
                sector_values[sector] += market_value

                rows.append({
                    'ticker': ticker,
                    'quantity': qty,
                    'buy_price': buy_price,
                    'current_price': current_price,
                    'value': market_value,
                    'gain_loss': gain_loss
                })
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")

    net_gain = total_value - total_cost
    return rows, ticker_values, sector_values, total_cost, net_gain

def create_distribution_charts(ticker_values, sector_values):
    try:
        labels = list(ticker_values.keys())
        sizes = list(ticker_values.values())
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Stock-wise Portfolio')
        plt.savefig('static/stock_distribution.png')
        plt.close()

        labels = list(sector_values.keys())
        sizes = list(sector_values.values())
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Sector-wise Portfolio')
        plt.savefig('static/sector_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error creating charts: {e}")

def fetch_fundamentals():
    rows = []
    for ticker in portfolio.keys():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            roe_value = info.get('returnOnEquity')
            if roe_value is None:
                roe_value = calculate_manual_roe(ticker)
            else:
                roe_value = roe_value * 100

            div_yield = info.get('dividendYield')
            pe = info.get('trailingPE')
            pb = info.get('priceToBook')
            debt_to_equity = info.get('debtToEquity')
            growth_rate = info.get('earningsQuarterlyGrowth')
            peg = info.get('pegRatio')

            if peg is None and pe and growth_rate and growth_rate != 0:
                peg = pe / (growth_rate * 100)

            rows.append({
                'Stock': ticker,
                'PEG': round(peg,2) if peg else 'N/A',
                'ROE': round(roe_value,2) if roe_value else 'N/A',
                'Debt/Equity': round(debt_to_equity/100,2) if debt_to_equity else 'N/A',
                'P/E': round(pe,2) if pe else 'N/A',
                'P/B': round(pb,2) if pb else 'N/A',
                'Dividend Yield': round(div_yield*100,2) if div_yield else 'N/A'
            })
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
    return pd.DataFrame(rows)

def evaluate_health(fund_df):
    verdicts = []
    for _, row in fund_df.iterrows():
        score = 0
        notes = []

        if isinstance(row['PEG'], (int, float)) and row['PEG'] <= 1.5: score +=1
        else: notes.append('High PEG')
        if isinstance(row['ROE'], (int, float)) and row['ROE'] >= 15: score +=1
        else: notes.append('Low ROE')
        if isinstance(row['Debt/Equity'], (int, float)) and row['Debt/Equity'] <= 0.5: score +=1
        else: notes.append('High Debt')
        if isinstance(row['P/E'], (int, float)) and row['P/E'] <= 15: score +=1
        else: notes.append('High P/E')
        if isinstance(row['P/B'], (int, float)) and row['P/B'] <= 1.5: score +=1
        else: notes.append('High P/B')
        if isinstance(row['Dividend Yield'], (int, float)) and row['Dividend Yield'] >= 2: score +=1

        verdict = '✅ Healthy' if score>=4 else '⚠️ Needs Attention' if score>=2 else '❌ Weak'
        verdicts.append({'Stock': row['Stock'], 'Score': score, 'Verdict': verdict, 'Notes': ", ".join(notes)})
    return pd.DataFrame(verdicts)

def create_health_chart(health_df):
    try:
        counts = health_df['Verdict'].value_counts()
        labels = [f"{v} ({counts[v]})" for v in counts.index]
        sizes = [counts[v] for v in counts.index]
        colors = ["#28a745" if "Healthy" in v else "#ffc107" if "Attention" in v else "#dc3545" for v in counts.index]

        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Portfolio Health Summary')
        plt.savefig('static/health_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error creating health chart: {e}")

# Routes
@app.route('/')
@app.route('/portfolio')
def portfolio_view():
    rows, ticker_vals, sector_vals, total_cost, net_gain = get_portfolio_data()
    return render_template('portfolio.html', rows=rows, total_cost=total_cost, net_gain=net_gain)

@app.route('/charts')
def charts_view():
    rows, ticker_vals, sector_vals, _, _ = get_portfolio_data()
    create_distribution_charts(ticker_vals, sector_vals)
    return render_template('charts.html')

@app.route('/fundamentals')
def fundamentals_view():
    fund_df = fetch_fundamentals()
    return render_template('fundamentals.html', fundamentals=fund_df.to_dict('records'))

@app.route('/health')
def health_view():
    fund_df = fetch_fundamentals()
    health_df = evaluate_health(fund_df)
    create_health_chart(health_df)
    return render_template('health.html', health=health_df.to_dict('records'))

# ✅ Load cache once at startup
load_cache()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
