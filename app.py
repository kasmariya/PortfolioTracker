from flask import Flask, render_template, url_for, redirect, request, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import requests
from flask import request, session, jsonify
from bs4 import BeautifulSoup
from babel.numbers import format_currency, format_decimal
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# Initialize Flask app and LoginManager
app = Flask(__name__)
app.secret_key = os.urandom(24)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class (validate login details with the stored data in credential.txt)
class User:
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def load_credentials():
        creds = {}
        with open("credential.txt", "r") as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    creds[key.strip()] = value.strip()
        return creds

    @staticmethod
    def get(id):
        creds = User.load_credentials()
        if id == 1:
            return User(id=1, username=creds['username'],
                        password_hash=generate_password_hash(creds['password']))
        return None

    @staticmethod
    def get_by_username(username):
        creds = User.load_credentials()
        if username == creds['username']:
            return User(id=1, username=creds['username'],
                        password_hash=generate_password_hash(creds['password']))
        return None

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


# Loading user into Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.get(int(user_id))

# Currentcy Format - with ₹ symbol
def format_inr(value):
    return format_currency(value, 'INR', locale='en_IN').replace(u'\xa0', u' ')  # To ensure spacing is correct

# For tables — no ₹ symbol
def format_inr_no_symbol(value):
        return format_decimal(value, locale='en_IN', format=u'#,##,##0.00')

# Register the function as a Jinja2 filter
app.jinja_env.filters['inr'] = format_inr      # Use in summaries
app.jinja_env.filters['inr_plain'] = format_inr_no_symbol  # Use in tables

#Stocks
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

#Mutual Funds
NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

mfportfolio = [
    {"folio": "5137979", "amfi_code": "119242", "scheme": "DSP ELSS Tax Saver Fund - Direct Plan - Growth", "invested": 285000.00, "balanced_units": 2666.080},
    {"folio": "16714214", "amfi_code": "119700", "scheme": "SBI Infrastructure Fund - Direct Plan - Growth", "invested": 202000.00, "balanced_units": 4072.150},
    {"folio": "1018550203", "amfi_code": "119514", "scheme": "Aditya Birla Sun Life Infrastructure Fund - PLAN - Growth - Direct Plan", "invested": 136222.31, "balanced_units": 1458.720},
    {"folio": "306653", "amfi_code": "131580", "scheme": "360 ONE Focused Equity Fund Direct Plan Growth", "invested": 97660.56, "balanced_units": 1942.370},
    {"folio": "3083144", "amfi_code": "151113", "scheme": "HSBC Value Fund - Direct Growth", "invested": 78000.00, "balanced_units": 679.130},
    {"folio": "7997047725", "amfi_code": "143783", "scheme": "Mirae Asset Healthcare Fund - Direct Plan", "invested": 71996.40, "balanced_units": 1803.005},
    {"folio": "599364433105", "amfi_code": "120731", "scheme": "UTI Transportation & Logistics Fund - Direct Plan", "invested": 72000.00, "balanced_units": 248.059},
    {"folio": "15387214", "amfi_code": "119769", "scheme": "Kotak India EQ Contra Fund - Direct Plan - Growth", "invested": 68000.00, "balanced_units": 397.700},
]



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
                #current_price = hist['Close'].iloc[-1]
                current_price = stock.fast_info["last_price"]
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


def get_eps_growth_next_year(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/analysis/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        # Look for all rows in all tables
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if cells and ticker in cells[0].text:
                    # EPS Growth section found
                    for r in rows:
                        c = r.find_all("td")
                        if c and ticker in c[0].text:
                            value = c[4].text.strip().replace("%", "")
                            return float(value)
        return None
    except Exception as e:
        print("Error fetching EPS growth:", e)
        return None

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
            growth_rate = get_eps_growth_next_year(ticker)
            peg = info.get('pegRatio')

            if peg is None and pe and growth_rate and growth_rate != 0:
                peg = pe / growth_rate

            rows.append({
                'Stock': ticker,
                'PEG': round(peg,2) if peg else 'N/A',
                'ROE': round(roe_value,2) if roe_value else 'N/A',
                'Debt/Equity': round(debt_to_equity/100,2) if debt_to_equity else 'N/A',
                'P/E': round(pe,2) if pe else 'N/A',
                'P/B': round(pb,2) if pb else 'N/A',
                'Dividend Yield': round(div_yield,2) if div_yield else 'N/A'
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
        else: notes.append('Low Dividend')

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

#Mutual funds
def update_mf_portfolio():
    response = requests.get(NAV_URL)
    nav_data = response.text.splitlines()
    nav_map = {}

    for line in nav_data:
        parts = line.split(';')
        if len(parts) >= 6:
            amfi_code = parts[0]
            nav = parts[4]
            nav_date = parts[5]
            if nav.replace('.', '', 1).isdigit():
                nav_map[amfi_code] = {"nav": float(nav), "date": nav_date}

    total_invested = 0
    total_market_value = 0

    for fund in mfportfolio:
        code = fund['amfi_code']
        if code in nav_map:
            fund['nav'] = nav_map[code]['nav']
            fund['nav_date'] = nav_map[code]['date']
            fund['market_value'] = round(fund['balanced_units'] * fund['nav'], 2)
            fund['gain_loss'] = round(fund['market_value'] - fund['invested'], 2)

            total_invested += fund['invested']
            total_market_value += fund['market_value']
        else:
            fund['nav'] = 0.0
            fund['nav_date'] = "N.A."
            fund['market_value'] = 0.0
            fund['gain_loss'] = 0.0

    total_gain_loss = round(total_market_value - total_invested, 2)
    return total_invested, total_market_value, total_gain_loss

# Routes - Stocks

# Sign-in and sign-out functionality
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_by_username(username)
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('networth_view'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


# For masking
@app.route('/update_mask', methods=['POST'])
def update_mask():
    data = request.get_json()
    # Extract 'mask' value from the JSON body
    mask = data.get('mask', False)

    # Store it in the session
    session['mask'] = mask

    # Respond with confirmation
    return jsonify({'success': True, 'mask': mask})

# ✅ Add Net Worth view - only accessible if logged in
@app.route('/')
@app.route("/networth")
@login_required
def networth_view():
    if 'mask' not in session:
        session['mask'] = True  # default to masking ON

    _, _, _, total_cost, net_gain = get_portfolio_data()
    stock_value = total_cost + net_gain

    _, mf_value, _ = update_mf_portfolio()

    my_net_worth = stock_value + mf_value

    indian_indices = {
    "SENSEX": "^BSESN",
    "NIFTY 50": "^NSEI",
    "NIFTY MIDCAP 100": "NIFTY_MIDCAP_100.NS",  
    "NIFTY SMALLCAP 100": "^CNXSC",             
    "NIFTY BANK": "^NSEBANK",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY IT": "^CNXIT",
    "NIFTY FMCG": "^CNXFMCG",
    "INDIA VIX": "^INDIAVIX"}

    index_data = {}
    for name, ticker in indian_indices.items():
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d")

            if len(hist) >= 2:
                latest = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = latest - prev
                percent = (change / prev) * 100

                index_data[name] = {
                    'value': round(latest, 2),
                    'change': round(percent, 2),
                    'direction': 'up' if change > 0 else 'down' if change < 0 else 'flat'
                }
            elif len(hist) == 1:
                latest = hist['Close'].iloc[-1]
                index_data[name] = {
                    'value': round(latest, 2),
                    'change': 0,
                    'direction': 'flat'
                }
            else:
                index_data[name] = {'value': 'N/A', 'change': 0, 'direction': 'flat'}

        except Exception as e:
            index_data[name] = {'value': 'N/A', 'change': 0, 'direction': 'flat'}


    return render_template("mynetworth.html",
                           total_value=format_inr(stock_value),
                           total_market_value=format_inr(mf_value),
                           my_net_worth=format_inr(my_net_worth),
                           stock_raw="{:.2f}".format(stock_value),
                           mf_raw="{:.2f}".format(mf_value),
                           mask=session['mask'],indices=index_data)


# Portfolio view 
@app.route('/portfolio')
@login_required
def portfolio_view():
    rows, ticker_vals, sector_vals, total_cost, net_gain = get_portfolio_data()
    total_value=total_cost+net_gain
    formatted_total_cost = format_inr(total_cost)
    formatted_total_value = format_inr(total_value)
    formatted_net_gain = format_inr(net_gain)

    return render_template('portfolio.html',rows=rows,total_cost=formatted_total_cost,total_value=formatted_total_value,
                           net_gain=formatted_net_gain)

# Routes for charts, fundamentals, and health - all require login
@app.route('/charts')
@login_required
def charts_view():
    rows, ticker_vals, sector_vals, _, _ = get_portfolio_data()
    create_distribution_charts(ticker_vals, sector_vals)
    return render_template('charts.html')

@app.route('/fundamentals')
@login_required
def fundamentals_view():
    fund_df = fetch_fundamentals()
    return render_template('fundamentals.html', fundamentals=fund_df.to_dict('records'))

@app.route('/health')
@login_required
def health_view():
    fund_df = fetch_fundamentals()
    health_df = evaluate_health(fund_df)
    create_health_chart(health_df)
    return render_template('health.html', health=health_df.to_dict('records'))

# Routes - Mutual funds
@app.route("/mfportfolio")
@login_required
def mf_portfolio_view():
    total_invested, total_market_value, total_gain_loss = update_mf_portfolio()
    return render_template("mfportfolio.html", portfolio=mfportfolio,
                           total_invested=format_inr(total_invested),
                           total_market_value=format_inr(total_market_value),
                           total_gain_loss=format_inr(total_gain_loss))

@app.route("/mfchart")
@login_required
def mf_chart_view():
    labels = [fund["scheme"] for fund in mfportfolio if fund["market_value"] > 0]
    values = [fund["market_value"] for fund in mfportfolio if fund["market_value"] > 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(values, labels=labels, startangle=90, autopct='%1.1f%%')
    plt.title("Portfolio Allocation", fontsize=16)
    plt.tight_layout()

    chart_path = "static/piechart.png"
    plt.savefig(chart_path)
    plt.close()
    return render_template("mfchart.html", chart_path=chart_path)


# ✅ Load cache once at startup
load_cache()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    #app.run(host='0.0.0.0', port=port)
    app.run(debug=True,port=port)
