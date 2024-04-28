import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the list of stocks
stocks = ['AAPL', 'GOOGL', 'NFLX', 'AMZN', 'TSLA']

# Download historical stock data for the most recent period (2019 to present)
data = yf.download(stocks, start="2019-01-01", end="2023-10-08")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Get user input for the number of user-specified portfolios
num_user_portfolios = int(input("Enter the number of portfolios with user-specified weights: "))

# Initialize results lists to store portfolio statistics
user_returns_list = []
user_stddev_list = []
user_weights_list = []

# Initialize variables to keep track of the best and worst portfolios
best_portfolio = None
best_portfolio_return = -float('inf')
worst_portfolio = None
worst_portfolio_return = float('inf')

# Get user input for portfolio weights
for i in range(num_user_portfolios):
    print(f"User-specified Portfolio {i + 1}:")
    weights = []
    for stock in stocks:
        weight = float(input(f"Enter weight for {stock}: "))
        weights.append(weight)
    
    # Normalize weights to ensure they sum up to 1
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    user_returns_list.append(portfolio_return)
    user_stddev_list.append(portfolio_stddev)
    user_weights_list.append(weights)
    
    # Check for best and worst portfolios
    if portfolio_return > best_portfolio_return:
        best_portfolio_return = portfolio_return
        best_portfolio = (weights, portfolio_return)
    if portfolio_return < worst_portfolio_return:
        worst_portfolio_return = portfolio_return
        worst_portfolio = (weights, portfolio_return)

# Convert lists to arrays for user-specified portfolios
user_returns_array = np.array(user_returns_list)
user_stddev_array = np.array(user_stddev_list)
user_weights_array = np.array(user_weights_list)

# Simulate random portfolios
num_random_portfolios = 100

# Initialize results lists to store portfolio statistics
random_returns_list = []
random_stddev_list = []

# Initialize variables to keep track of the best random portfolio
best_random_portfolio = None
best_random_portfolio_return = -float('inf')

# Simulate random portfolios
for i in range(num_random_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    random_returns_list.append(portfolio_return)
    random_stddev_list.append(portfolio_stddev)
    
    # Check for the best random portfolio
    if portfolio_return > best_random_portfolio_return:
        best_random_portfolio_return = portfolio_return
        best_random_portfolio = (weights, portfolio_return)

# Convert lists to arrays for random portfolios
random_returns_array = np.array(random_returns_list)
random_stddev_array = np.array(random_stddev_list)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(user_stddev_array, user_returns_array, label='User-specified Portfolios', c='green', marker='o')
plt.scatter(random_stddev_array, random_returns_array, label='Random Portfolios', c='blue', marker='x')

# Highlight the best and worst user-specified portfolios in red
best_weights, best_return = best_portfolio
worst_weights, worst_return = worst_portfolio
plt.scatter(np.sqrt(np.dot(best_weights, np.dot(cov_matrix, best_weights))) * np.sqrt(252), best_return, c='red', marker='o', s=100, label='Best User Portfolio')
plt.scatter(np.sqrt(np.dot(worst_weights, np.dot(cov_matrix, worst_weights))) * np.sqrt(252), worst_return, c='red', marker='o', s=100, label='Worst User Portfolio')

# Label the best and worst portfolios with their weights and number of stocks
plt.text(np.sqrt(np.dot(best_weights, np.dot(cov_matrix, best_weights))) * np.sqrt(252), best_return, f'Best User Portfolio\nWeights: {best_weights}\nStocks: {np.sum(np.array(best_weights) > 0)}', fontsize=10, ha='right')
plt.text(np.sqrt(np.dot(worst_weights, np.dot(cov_matrix, worst_weights))) * np.sqrt(252), worst_return, f'Worst User Portfolio\nWeights: {worst_weights}\nStocks: {np.sum(np.array(worst_weights) > 0)}', fontsize=10, ha='right')

# Highlight the best random portfolio in red with a cross
best_random_weights, best_random_return = best_random_portfolio
plt.scatter(np.sqrt(np.dot(best_random_weights, np.dot(cov_matrix, best_random_weights))) * np.sqrt(252), best_random_return, c='red', marker='x', s=100, label='Best Random Portfolio')

# Label the best random portfolio with its weights and number of stocks
plt.text(np.sqrt(np.dot(best_random_weights, np.dot(cov_matrix, best_random_weights))) * np.sqrt(252), best_random_return, f'Best Random Portfolio\nWeights: {best_random_weights}\nStocks: {np.sum(np.array(best_random_weights) > 0)}', fontsize=10, ha='right')

plt.title('Efficient Frontier')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.legend()
plt.show()


# Assuming you have predicted returns stored in a variable named 'predicted_returns'

# Calculate absolute errors
absolute_errors = abs(predicted_returns - returns.mean())

# Calculate mean absolute error
mean_absolute_error = absolute_errors.mean()

print(f"Mean Absolute Error: {mean_absolute_error}")


