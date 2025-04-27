from flask import Flask, render_template, send_file
import yfinance as yf
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Updated portfolio with stock symbols, quantities, and buy price
portfolio = {
    'TATAMOTORS.NS': (600, 740.42, 'Automobile'),
    'JIOFIN.NS': (1550, 228.49, 'Finance'),
    'TCS.NS': (79, 3472.58, 'Technology'),
    'TATAPOWER.NS': (550, 318.05, 'Energy'),
    'BEL.NS': (580, 249.54, 'Defense'),         
    'IRCTC.NS': (225, 715.00, 'Transport'),
    'TITAN.NS': (50, 3016.05, 'Retail'),
    'MOTHERSON.NS': (1000, 124.14, 'Automobile'),
    'HINDUNILVR.NS': (50, 2178.46, 'FMCG'),  
    'BAJAJHFL.NS': (690, 114.77, 'Finance')     
}

# Function to fetch live price of stock from Yahoo Finance
def fetch_live_price(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    todays_data = ticker.history(period='1d')
    if not todays_data.empty:
        return round(todays_data['Close'][0], 2)
    else:
        return None

# Function to calculate updated portfolio with added columns
def get_portfolio():
    updated_portfolio = []
    for stock, (quantity, buy_price, sector) in portfolio.items():
        current_price = fetch_live_price(stock)
        if current_price is not None:
            value = round(quantity * current_price, 2)
            gain_loss = round(value - (quantity * buy_price), 2)
        else:
            current_price = gain_loss = value = None

        updated_portfolio.append({
            'ticker': stock.replace(".NS", ""),
            'quantity': quantity,
            'buy_price': buy_price,
            'current_price': current_price,
            'value': value,
            'gain_loss': gain_loss,
            'sector': sector
        })
    return updated_portfolio

# Function to create the pie chart (Stock-wise)
def generate_stock_chart():
    data = get_portfolio()
    labels = [item['ticker'] for item in data]
    sizes = [item['value'] for item in data]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    ax.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# Function to create the pie chart (Sector-wise)
def generate_sector_chart():
    # Aggregate values by sector
    sector_values = {}
    for stock, (quantity, buy_price, sector) in portfolio.items():
        current_price = fetch_live_price(stock)
        if current_price is not None:
            value = quantity * current_price
            if sector in sector_values:
                sector_values[sector] += value
            else:
                sector_values[sector] = value

    labels = list(sector_values.keys())
    sizes = list(sector_values.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    ax.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# Route for portfolio
@app.route('/')
def portfolio_view():
    data = get_portfolio()
    return render_template('portfolio.html', portfolio=data)

# Route for stock-wise chart
@app.route('/stock-chart.png')
def stock_chart():
    img = generate_stock_chart()
    return send_file(img, mimetype='image/png')

# Route for sector-wise chart
@app.route('/sector-chart.png')
def sector_chart():
    img = generate_sector_chart()
    return send_file(img, mimetype='image/png')

# Route for health score view
@app.route('/health')
def health_score():
    score = 85  # Example score
    return render_template('health.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)
