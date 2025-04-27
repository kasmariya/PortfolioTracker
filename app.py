from flask import Flask, render_template, send_file
import yfinance as yf
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Updated portfolio with stock symbols and quantities
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

# Function to fetch live price of stock from Yahoo Finance
def fetch_live_price(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    todays_data = ticker.history(period='1d')
    if not todays_data.empty:
        return round(todays_data['Close'][0], 2)
    else:
        return None

# Function to calculate updated portfolio value
def get_portfolio():
    updated_portfolio = []
    for stock, (quantity, price) in portfolio.items():
        updated_portfolio.append({
            'stock': stock.replace(".NS", ""),
            'quantity': quantity,
            'price': price,
            'total_value': round(quantity * price, 2)
        })
    return updated_portfolio

# Route for portfolio
@app.route('/')
def portfolio_view():
    data = get_portfolio()
    return render_template('portfolio.html', portfolio=data)

# Route for chart view
@app.route('/chart')
def chart_view():
    return render_template('chart.html')

# Route for health score view
@app.route('/health')
def health_score():
    score = 85  # Example score
    return render_template('health.html', score=score)

# Route for portfolio pie chart
@app.route('/chart.png')
def chart():
    data = get_portfolio()
    labels = [item['stock'] for item in data]
    sizes = [item['total_value'] for item in data]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    ax.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
