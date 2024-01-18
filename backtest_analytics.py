import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go
import empyrical as ep

class BacktestAnalytics:
    def __init__(self, portfolio, benchmark, portfolio_stocks, starting_cash, round_digits=2):
        self.portfolio_balance = portfolio
        self.benchmark = benchmark
        self.portfolio_stocks = portfolio_stocks
        self.starting_cash = starting_cash
        self.round_digits = round_digits

        self.start_value = self.portfolio_balance['balance'].iloc[0]
        self.end_value = self.portfolio_balance['balance'].iloc[-1]
        self.periods = len(self.portfolio_balance) / 252
        print('   start/end: ',self.start_value, '  ', self.end_value, '  ', self.periods )
        print('   period len: ', self.periods)
        
        self.returns = self.portfolio_balance['balance'].pct_change().dropna()
        self.benchmark_returns = self.benchmark['Adj Close'].pct_change().dropna()
        
        # benchmark calculations
        self.benchmark_cagr = self.calculate_cagr(self.benchmark['Adj Close'].iloc[0], self.benchmark['Adj Close'].iloc[-1], self.periods)
        self.benchmark_sharpe = self.calculate_sharpe_ratio(self.benchmark_returns)
    
    def calculate_cagr(self, start_value, end_value, periods):
        cagr = (end_value / start_value) ** (1/periods) - 1
        return round(cagr * 100, self.round_digits)

    def calculate_total_return(self):
        total_return = (self.end_value - self.start_value) / self.start_value
        return round(total_return * 100, self.round_digits)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0):
        # Convert daily returns to annualized returns
        annualized_return = np.mean(returns) * 252
        annualized_volatility = np.std(returns) * np.sqrt(252)

        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        return round(sharpe_ratio, self.round_digits)

    def calculate_max_drawdown(self):
        rolling_max = self.portfolio_balance['balance'].cummax()
        drawdown = (self.portfolio_balance['balance'] - rolling_max) / rolling_max
        return round(drawdown.min() * 100, self.round_digits)

    def calculate_ending_cash(self):
        return round(self.starting_cash * (1 + self.calculate_total_return() / 100), self.round_digits)

    def calculate_benchmark_ending_cash(self):
        return round(self.starting_cash * (1 + self.benchmark_cagr / 100), self.round_digits)

    def print_results(self):
        print(f"Starting cash: {self.starting_cash}")
        print(f"Ending cash: {self.calculate_ending_cash()}")
        print(f"CAGR: {self.calculate_cagr(self.start_value, self.end_value, self.periods)}%")
        print(f"Total return: {self.calculate_total_return()}%")
        print(f"Sharpe ratio: {self.calculate_sharpe_ratio(self.returns)}")
        print(f"Max drawdown: {self.calculate_max_drawdown()}%")
        print(f"Benchmark CAGR: {self.benchmark_cagr}%")
        print(f"Benchmark Sharpe ratio: {self.benchmark_sharpe}")
        print(f"Benchmark ending cash: {self.calculate_benchmark_ending_cash()}")

    def plot_data(self):
        # Normalize the data
        normalized_portfolio = self.portfolio_balance['balance'] / self.portfolio_balance['balance'].iloc[0] * self.starting_cash
        normalized_benchmark = self.benchmark['Adj Close'] / self.benchmark['Adj Close'].iloc[0] * self.starting_cash

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.portfolio_balance['date'], y=normalized_portfolio,
                        mode='lines',
                        name='Portfolio'))

        fig.add_trace(go.Scatter(x=self.benchmark.index, y=normalized_benchmark,
                        mode='lines',
                        name='Benchmark (SPY)'))

        fig.show()



class PerformanceAnalytics:
    def __init__(self, portfolio_balance, benchmark_data, starting_cash=1000):
        self.starting_cash = starting_cash

        # Initialize portfolio balance and benchmark data as pandas Series with date as index
        self.balance = portfolio_balance.set_index('date')['balance']
        # Check if the index of benchmark_data is already set to 'Date'
        if benchmark_data.index.name != 'Date':
            self.benchmark_data = benchmark_data.set_index('Date')['Adj Close']
        else:
            self.benchmark_data = benchmark_data['Adj Close']

        # Convert index to datetime if it's not already
        self.balance.index = pd.to_datetime(self.balance.index)
        self.benchmark_data.index = pd.to_datetime(self.benchmark_data.index)

        # Make sure both Series cover the same date range
        self.start_date = max(self.balance.index[0], self.benchmark_data.index[0])
        self.end_date = min(self.balance.index[-1], self.benchmark_data.index[-1])
        self.balance = self.balance[self.start_date:self.end_date]
        self.benchmark_data = self.benchmark_data[self.start_date:self.end_date]

        # Normalize benchmark data to the same starting cash
        self.benchmark_data_normalized = self.benchmark_data / self.benchmark_data[0] * starting_cash

        # Calculate returns
        self.returns = self.balance.pct_change().dropna()
        self.benchmark_returns = self.benchmark_data_normalized.pct_change().dropna()

        # Calculate the number of years
        self.years = (self.returns.index[-1] - self.returns.index[0]).days / 365.25
        self.benchmark_years = (self.benchmark_returns.index[-1] - self.benchmark_returns.index[0]).days / 365.25

    def print_results(self):
        # Calculate metrics
        ending_cash = self.balance.iloc[-1]
        cagr = ((ending_cash / self.starting_cash) ** (1/self.years) - 1) if self.years else np.nan
        total_return = ep.cum_returns_final(self.returns)
        sharpe_ratio = ep.sharpe_ratio(self.returns)
        max_drawdown = ep.max_drawdown(self.returns)
        sortino_ratio = ep.sortino_ratio(self.returns)
        downside_risk = ep.downside_risk(self.returns)
        var = ep.value_at_risk(self.returns)
        cvar = ep.conditional_value_at_risk(self.returns)
        alpha, beta = ep.alpha_beta(self.returns, self.benchmark_returns)

        # Calculate benchmark metrics
        benchmark_ending_cash = self.benchmark_data_normalized.iloc[-1]
        benchmark_cagr = ((benchmark_ending_cash / self.starting_cash) ** (1/self.benchmark_years) - 1) if self.benchmark_years else np.nan
        benchmark_sharpe_ratio = ep.sharpe_ratio(self.benchmark_returns)
        benchmark_sortino_ratio = ep.sortino_ratio(self.benchmark_returns)

        # Calculate differences
        sharpe_difference = sharpe_ratio - benchmark_sharpe_ratio
        sortino_difference = sortino_ratio - benchmark_sortino_ratio

        # Print the metrics
        print('Starting cash:', self.starting_cash)
        print('Ending cash:', round(ending_cash,2))
        print('CAGR: {:.2%}'.format(cagr))
        print('Total return: {:.2%}'.format(total_return))
        print('Sharpe ratio:', round(sharpe_ratio, 2))
        print('Max drawdown: {:.2%}'.format(max_drawdown))
        print('Sortino ratio:', round(sortino_ratio, 2))
        print('Downside risk:', round(downside_risk, 2))
        print('Value at risk:', round(var, 2))
        print('Conditional value at risk:', round(cvar, 2))
        print('Alpha:', round(alpha, 2))
        print('Beta:', round(beta, 2))
        print('Benchmark CAGR: {:.2%}'.format(benchmark_cagr))
        print('Benchmark Sharpe ratio:', round(benchmark_sharpe_ratio, 2))
        print('Benchmark Sortino ratio:', round(benchmark_sortino_ratio, 2))
        print('Benchmark ending cash:', round(benchmark_ending_cash, 2))
        print('Sharpe Ratio Difference: {:.2f}'.format(sharpe_difference))
        print('Sortino Ratio Difference: {:.2f}'.format(sortino_difference))
