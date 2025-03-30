import yfinance as yf
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

#Cleans the stock data
def clean_stock_data(df):
    df.reset_index(inplace=True)
    df.columns = df.columns.to_flat_index()
    df.columns = ["_".join(col) for col in df.columns]
    df.rename(columns={"Date_": "Date"}, inplace=True)
    return df

#Top 20 and bottom 20 stocks on the S&P500 in accordance to market cap
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "GOOGL", "META", "PG", "TSLA", "AVGO",
          "LLY", "JPM", "WMT", "V", "XOM", "MA", "UNH", "NFLX", "ORCL", "COST", 
          
          "HSIC", "MOS",
          "HAS", "FRT", "CRL", "MKTX", "GNRC", "HII", "MTCH", "ENPH", "PARA", "APA", "LW", "MHK",
          "IVZ", "BWA", "CE", "TFX", "CZR", "FMC"]

stock_data = yf.download(tickers, start="2022-01-01", end="2025-01-01")[['Close']]
stock_data = clean_stock_data(stock_data)

stock_data = stock_data.dropna(axis=1)
stock_data.set_index('Date', inplace=True)


daily_returns = stock_data.pct_change().dropna()
corr_matrix = daily_returns.corr()


G = nx.Graph()
threshold = 0.40
for i in corr_matrix.columns:
    for j in corr_matrix.columns:
        if i != j and corr_matrix.loc[i, j] > threshold:
            G.add_edge(i, j, weight = corr_matrix.loc[i,j])


degree_centrality = nx.degree_centrality(G)
centrality_closeness = nx.closeness_centrality(G)
centrality_pagerank = nx.pagerank(G)

centrality_df = pd.DataFrame({
    'Degree Centrality': pd.Series(degree_centrality),
    'Closeness Centrality': pd.Series(centrality_closeness),
    'PageRank': pd.Series(centrality_pagerank)
}).sort_values(by='PageRank', ascending = False)


#List for most influential stocks
influential_stocks_df = pd.DataFrame({
    'PageRank': pd.Series(centrality_pagerank),
    'Number of Connections': pd.Series(dict(G.degree()))
}).sort_values(by='PageRank', ascending=False)
#Includes pagerank and the number of connection that the stock has
print("Top 10 Most Influential Stocks (by PageRank):\n")
print(influential_stocks_df.head(10).to_string())

#Top 3 nodes
top_nodes = influential_stocks_df.head(3).index.tolist()
node_color = ['green' if node in top_nodes else 'skyblue' for node in G.nodes]



#Plots the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=20, k = 1.2)
node_sizes = [centrality_pagerank[node] * 6000 for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=0.5)

plt.title("S&P 500 Stock Influence")
plt.axis('off')
plt.show()







