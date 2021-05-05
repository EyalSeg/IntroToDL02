import math
import pandas as pd
import matplotlib.pyplot as plt


def daily_max_stock(stocks, title="Stocks"):
    sd = stocks['date'][:round(len(stocks) * .1)]
    sdv = stocks['dvolume'][:round(len(stocks) * .1)]

    plt.plot(range(len(sdv)), sdv)
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Max Dollar Volume")
    plt.show()

    print(f"First Date is: {sd.values[0]}\nLast Date is: {sd.values[math.floor(len(stocks) * .1)]}")


def plot_max_stocks():
    stocks = pd.read_csv('../../data/cache/S&P500.csv')

    google_stocks = stocks[stocks['symbol'] == 'GOOGL']
    amazon_stocks = stocks[stocks['symbol'] == 'AMZN']

    daily_max_stock(google_stocks, "Google Stocks")
    daily_max_stock(amazon_stocks, "Amazon Stocks")


if __name__ == "__main__":
    plot_max_stocks()
