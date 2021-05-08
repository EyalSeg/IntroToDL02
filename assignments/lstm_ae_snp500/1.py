import math
import pandas as pd
import matplotlib.pyplot as plt


# Fetch "Date" and "High" Entries out of The Stocks Data Set
# Plot the "High" at the Y-axis, and the Length of "High" as the X-axis
# To indicate the total days.
# Print the First Date, and the last date (out of "Date" Entries of the Stocks Data Set).
def daily_max_stock(stocks, title="Stocks"):
    sd = stocks['date'][:round(len(stocks) * .1)]
    sdv = stocks['high'][:round(len(stocks) * .1)]

    plt.plot(range(len(sdv)), sdv)
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Max Dollar Volume")
    plt.show()

    print(f"First Date is: {sd.values[0]}\nLast Date is: {sd.values[math.floor(len(stocks) * .1)]}")


# Fetch The S&P00 Data Set
# Fetch Google Stocks, and Amazon Stocks
# Plot the "Daily-max" Stock of each Symbol
# using "daily_max_stock" variable
def plot_max_stocks():
    stocks = pd.read_csv('../../data/cache/S&P500.csv')

    google_stocks = stocks[stocks['symbol'] == 'GOOGL']
    amazon_stocks = stocks[stocks['symbol'] == 'AMZN']

    daily_max_stock(google_stocks, "Google Stocks")
    daily_max_stock(amazon_stocks, "Amazon Stocks")


if __name__ == "__main__":
    plot_max_stocks()