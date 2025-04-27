import os
import pickle
import yfinance as yf
import pandas as pd
from flask import Flask, render_template
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Global cache dictionary
financials_cache = {}

# File where cache will be saved
CACHE_FILE = 'financials_cache.pkl'

# Load cache if exists
def load_cache():
    global financials_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            financials_cache = pickle.load(f)
        print("✅ Loaded financials cache from file.")

# Save cache to file
def save_cache():
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(financials_cache, f)
    print("💾 Saved financials cache to file.")

# Portfolio data
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

def calculate_manual_roe(ticker):
    try:
        if ticker not in financials_cache:
            stock = yf.Ticker(ticker)
            financials_cache[ticker] = {
                'income_stmt': stock.financials,
                'balance_sheet': stock.balance_sheet
            }
            save_cache()  # Save after adding new stock

        income_stmt = financials_cache[ticker]['income_stmt']
        balance_sheet = financials_cache[ticker]['balance_sheet']

        net_income = None
        shareholder_equity = None

        # Try multiple possible field names for Net Income
        for possible_net_income in ['Net Income', 'NetIncome', 'Net Income Common Stockholders']:
            if possible_net_income in income_stmt.index:
                net_income = income_stmt.loc[possible_net_income].iloc[0]
                break
        
        # Try multiple possible field names for Equity
        for possible_equity in ['Total Stockholder Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']:
            if possible_equity in balance_sheet.index:
                shareholder_equity = balance_sheet.loc[possible_equity].iloc[0]
                break

        if net_income is None or shareholder_equity is None or shareholder_equity == 0:
            return None
        
        roe = (net_income / shareholder_equity) * 100
        return round(roe, 2)

    except Exception as e:
        print(f"⚠️ Error calculating manual ROE for {ticker}: {e}")
        return None

def fetch_fundamentals(portfolio):
    rows = []
    for ticker in portfolio.keys():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get necessary values
            roe_value = info.get('returnOnEquity', None)
            div_value = info.get('dividendYield', None)
            pe_value = info.get('trailingPE', None)
            growth_rate = info.get("earningsQuarterlyGrowth")  # Quarterly EPS growth
            peg_value = info.get('pegRatio', None)

            # If PEG not available, try to calculate
            if peg_value is None and pe_value is not None and growth_rate is not None and growth_rate != 0:
                peg_value = pe_value / (growth_rate * 100)  # growth_rate is like 0.15 → 15%

            # 🌟 If ROE is missing, calculate manually
            if roe_value is None:
                roe_value = calculate_manual_roe(ticker)
            else:
                roe_value = roe_value * 100  # Convert from decimal to %

            row = {
                "Stock": ticker,
                "PEG": round(peg_value, 2) if peg_value is not None else 'N/A',
                "ROE": round(roe_value, 2) if roe_value is not None else 'N/A',
                "Debt/Equity": round(info['debtToEquity'] / 100, 2) if info.get('debtToEquity') is not None else 'N/A',
                "P/E": round(pe_value, 2) if pe_value is not None else 'N/A',
                "P/B": round(info['priceToBook'], 2) if info.get('priceToBook') is not None else 'N/A',
                "Dividend Yield": round(div_value * 100, 2) if div_value is not None else 'N/A'
            }
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Error fetching data for {ticker}: {e}")
    return pd.DataFrame(rows)

def is_number(x):
    return isinstance(x, (int, float))

def evaluate_portfolio_health(fundamentals_df):
    verdicts = []
    for idx, row in fundamentals_df.iterrows():
        name = row['Stock']
        roe = row.get('ROE')
        debt_eq = row.get('Debt/Equity')
        pe = row.get('P/E')
        pb = row.get('P/B')
        peg = row.get('PEG')
        div = row.get('Dividend Yield')

        score = 0
        notes = []

        if is_number(peg) and peg <= 1.5:
            score += 1
        else:
            notes.append("High PEG")

        if is_number(roe) and roe >= 15:
            score += 1
        else:
            notes.append("Low ROE")

        if is_number(debt_eq) and debt_eq <= 0.5:
            score += 1
        else:
            notes.append("High Debt")

        if is_number(pe) and pe <= 15:
            score += 1
        else:
            notes.append("High P/E")

        if is_number(pb) and pb <= 1.5:
            score += 1
        else:
            notes.append("High P/B")

        if is_number(div) and div >= 2:
            score += 1

        verdict = "✅ Healthy" if score >= 4 else "⚠️ Needs Attention" if score >= 2 else "❌ Weak Fundamentals"

        verdicts.append({
            "Stock": name,
            "PEG": peg,
            "ROE": roe,
            "Debt/Equity": debt_eq,
            "P/E": pe,
            "P/B": pb,
            "Dividend Yield": div,
            "Score": score,
            "Verdict": verdict,
            "Notes": ", ".join(notes)
        })

    return pd.DataFrame(verdicts)

def create_health_pie_chart(health_df):
    # Color map
    color_map = {
        "✅ Healthy": '#28a745',        # Green
        "⚠️ Needs Attention": '#ffc107', # Yellow
        "❌ Weak Fundamentals": '#dc3545' # Red
    }

    # Group stocks by Verdict
    grouped = health_df.groupby('Verdict')['Stock'].apply(list)

    labels = []
    sizes = []
    colors = []

    for verdict, stocks in grouped.items():
        stock_list = ", ".join(stocks)  # Join stock names
        label = f"{verdict} ({len(stocks)})\n{stock_list}"  # Verdict + count + stocks
        labels.append(label)
        sizes.append(len(stocks))
        colors.append(color_map.get(verdict, '#6c757d'))

    # Plot
    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 10},
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title('🧠 Portfolio Health Summary', fontsize=18)
    plt.axis('equal')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

@app.route('/')
def portfolio_view():
    fundamentals_df = fetch_fundamentals(portfolio)
    health_df = evaluate_portfolio_health(fundamentals_df)
    return render_template('portfolio.html', portfolio=portfolio, health_df=health_df)

@app.route('/health')
def health_score_view():
    fundamentals_df = fetch_fundamentals(portfolio)
    health_df = evaluate_portfolio_health(fundamentals_df)
    img = create_health_pie_chart(health_df)
    return render_template('health.html', health_df=health_df, img=img)

if __name__ == '__main__':
    load_cache()  # Load cache if available
    app.run(debug=True)
