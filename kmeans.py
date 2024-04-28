import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Define the list of stocks
stocks = ['AAPL', 'GOOGL', 'NFLX', 'AMZN', 'TSLA']

# Fetch historical data for the stocks
data = yf.download(stocks, start='2019-01-01', end='2023-01-01')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate expected returns and standard deviations
expected_returns = returns.mean()
std_devs = returns.std()

# Create a DataFrame to store expected returns and standard deviations
risk_return_df = pd.DataFrame({'Expected Return': expected_returns, 'Standard Deviation': std_devs})

# Generate 100 sets of random weights for the stocks
random_weights = [np.random.dirichlet(np.ones(len(stocks)), size=1).tolist()[0] for _ in range(100)]

# User input for number of clusters
num_clusters = int(input("Enter the number of clusters (n): "))

# Combine randomly generated and user-input portfolios
all_portfolios = random_weights

# Calculate expected return and standard deviation for each portfolio
portfolio_returns = []
portfolio_std_devs = []

for weights in all_portfolios:
    expected_portfolio_return = np.dot(expected_returns, weights)
    portfolio_std_dev = np.sqrt(np.dot(weights, np.dot(returns.cov(), weights)))
    portfolio_returns.append(expected_portfolio_return)
    portfolio_std_devs.append(portfolio_std_dev)

# Create a DataFrame to store expected returns and standard deviations of portfolios
portfolios_df = pd.DataFrame({'Expected Return': portfolio_returns, 'Standard Deviation': portfolio_std_devs})

# Perform K-means clustering to group portfolios
X = portfolios_df.values
kmeans_portfolios = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(X)
portfolios_df['Cluster'] = kmeans_portfolios.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))

for i in range(num_clusters):
    cluster_data = portfolios_df[portfolios_df['Cluster'] == i]
    plt.scatter(cluster_data['Standard Deviation'], cluster_data['Expected Return'], label=f'Cluster {i}')

plt.title(f'Risk-Return Clusters for {num_clusters} Portfolios')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()

# Print the portfolios in each cluster
for i in range(num_clusters):
    cluster_portfolios = portfolios_df[portfolios_df['Cluster'] == i].index
    print(f'Portfolios in Cluster {i}: {", ".join(map(str, cluster_portfolios))}')