from flask import Flask, render_template, send_file
import yfinance as yf
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

portfolio_data = [
    {"stock": "TATAMOTORS.NS", "quantity": 10},
    {"stock": "TATAPOWER.NS", "quantity": 15},
]

def fetch_live_price(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    todays_data = ticker.history(period='1d')
    if not todays_data.empty:
        return round(todays_data['Close'][0], 2)
    else:
        return None

def get_portfolio():
    updated_portfolio = []
    for item in portfolio_data:
        price = fetch_live_price(item["stock"])
        total_value = round(item["quantity"] * price, 2) if price else 0
        updated_portfolio.append({
            "stock": item["stock"].replace(".NS", ""),
            "quantity": item["quantity"],
            "price": price if price else "N/A",
            "total_value": total_value
        })
    return updated_portfolio

@app.route('/')
def portfolio():
    data = get_portfolio()
    return render_template('portfolio.html', portfolio=data)

@app.route('/chart')
def chart_view():
    return render_template('chart.html')

@app.route('/health')
def health_score():
    score = 85
    return render_template('health.html', score=score)

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