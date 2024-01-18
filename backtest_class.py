import pandas as pd
import sys
import os
from db import db_util 
from db.db_util import Database
from datetime import datetime, timedelta
import numpy as np
import random
import riskfolio as rp
import time
import ta


class GenericBacktestMethods():
    def __init__(self, start_date, end_date, rebalance_period_days, db_connection, initial_cash):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_period_days = rebalance_period_days
        self.db_connection = db_connection
        self.portfolio = pd.DataFrame(columns=['date', 'symbol', 'weight', 'price', 'value', 'shares'])
        self.portfolio_balance = pd.DataFrame(columns=['date', 'balance'])
        self.last_rebalance_date = None
        self.initial_cash = initial_cash
        self.allocations = {}
        self.liquid_cash = initial_cash  # new variable to keep track of available cash
        self.spy_data = None
        self.db = Database()
    
    def generate_rebalance_dates(self, start_date, end_date, rebalance_period):
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=f'{rebalance_period}D')
        return rebalance_dates

    def run(self):
        self.spy_data = self.db.get_price('SPY', self.start_date, self.end_date, table='benchmark')
        trading_days = pd.to_datetime(self.spy_data.index).tolist()
        print('trading days: ',len(trading_days), '    '+15*'+++---') #divider
        for trading_day in trading_days:
            start_time = time.time()  # Start timing here
            self.on_bar(trading_day)
            self.record_portfolio(trading_day)
            elapsed_time = time.time() - start_time  # Get the elapsed time
            elapsed_minutes = round(elapsed_time / 60, 2)  # Convert to minutes and round to 2 decimal places
            trading_day_str = trading_day.strftime("%Y-%m-%d")
            # print(f'The simulation for {trading_day_str} took {elapsed_minutes} minutes.')
        self.db.close()
    
    def record_portfolio(self, current_date):
        daily_balance = 0  + self.liquid_cash
        symbols = list(self.allocations.keys())
        prices = self.db.get_prices(symbols, current_date, today_only=True)  # Get prices from db
        if prices is None:  # Continue if prices is None
            print("No prices returned from db.")
            return
        for symbol, price_data in prices.items():
            weight = self.allocations[symbol]['weight']
            shares = self.allocations[symbol]['shares']
            price = price_data.get(str(current_date), None)
            if price is None:  # Continue if the price for the current date is not found
                print(f"No price for {symbol} on {current_date}")
                continue
            value = shares * price
            daily_balance += value
            self.portfolio = pd.concat([self.portfolio, pd.DataFrame([{
                'date': current_date,
                'symbol': symbol,
                'weight': weight,
                'price': price,
                'value': value,
                'shares': shares
            }])], ignore_index=True)
        self.portfolio_balance = pd.concat([self.portfolio_balance, pd.DataFrame([{
            'date': current_date,
            'balance': daily_balance
        }])], ignore_index=True)
        print('    balance:  $', round(daily_balance,2) ,'  ',current_date.strftime("%Y-%m-%d"), )
        self.available_cash = daily_balance
        return

class Backtest(GenericBacktestMethods):
    def __init__(self, start_date, end_date, rebalance_period_days, db_connection, initial_cash, lookback, uppersht, upperlng, long_short_budget, max_long_allocation, max_short_allocation):
        super().__init__(start_date, end_date, rebalance_period_days, db_connection, initial_cash, )
        self.db = Database('db/stocks.db')
        self.db.connect()  # Open the database connection when initializing Backtest
        self.lookback = lookback
        self.portfolio_allocations = pd.DataFrame(columns=['date', 'symbol', 'weight'])
        self.uppersht = uppersht
        self.upperlng = upperlng
        self.long_short_budget = long_short_budget
        self.last_called = None
        self.max_long_allo = max_long_allocation
        self.max_short_allo = max_short_allocation


    def on_bar(self, current_date):
        if self.last_rebalance_date is None or (current_date - self.last_rebalance_date).days >= self.rebalance_period_days:
            print(f'  >>  Rebalancing on: ', current_date.strftime('%Y-%m-%d'), )
            self.rebalance(current_date)
            self.last_rebalance_date = current_date
        return
    
    def liquid_portfolio_worth(self, current_date, liquidate=False):
        x = self.db.get_price('MSFT', current_date, today_only=True)
        x = x.iloc[0]['Adj Close']
        portfolio_stock_cash_equivilant = sum([(self.db.get_price(symbol, current_date, today_only=True)).iloc[0]['Adj Close'] * self.allocations[symbol]['shares'] for symbol in self.allocations.keys()])
        print('portfolio_stock_cash_equivilant ', portfolio_stock_cash_equivilant)
        portfolio_cash_equivilant = portfolio_stock_cash_equivilant + self.liquid_cash
        if liquidate: self.allocations = {}
        return portfolio_cash_equivilant

    def rebalance(self, current_date):
        start_time = datetime.now() # Start the timer
        if self.last_called is not None:
            minutes_since_last_call = (time.time() - self.last_called) / 60
            print(f"Time between rebalances: {round(minutes_since_last_call, 2)} minutes")
        self.liquid_cash = self.liquid_portfolio_worth(current_date, liquidate=True)  #get total cash we have, basically sell everything
        new_weights = self.generate_allocation_weights(current_date)
        for symbol, weight in new_weights.items():
            self.portfolio_allocations = pd.concat([self.portfolio_allocations, pd.DataFrame([{
                'date': current_date,
                'symbol': symbol,
                'weight': round(weight,4)
            }])], ignore_index=True)
        #buy stocks
        portfolio_total_cash = self.liquid_cash
        for symbol, weight in new_weights.items():
            price = self.db.get_price(symbol, current_date, today_only=True)
            price = price.iloc[0]['Adj Close']
            cash = portfolio_total_cash * weight
            self.allocations[symbol] = {
                'weight': weight,
                'shares': cash / price
            }
            self.liquid_cash = self.liquid_cash - cash
        self.liquid_cash = round(self.liquid_cash,8)
        allocation_weights = {key: round(value['weight'], 8) for key, value in self.allocations.items()}
        df = pd.DataFrame([allocation_weights])
        df = df.sort_values(by=0, axis=1, ascending=False)
        self.last_called = time.time()
        end_time = datetime.now()
        elapsed_time = round((end_time - start_time).total_seconds() / 60 ,2)
        print(f'The function took {elapsed_time} minutes to run.')
        return self.allocations

    def generate_allocation_weights(self, current_date):
        next_rebalance_date = current_date + timedelta(days=self.rebalance_period_days + 1)
        lookback_rebalance_date = current_date - timedelta(days=self.lookback + 1)
        spy_data = self.db.get_price('SPY', lookback_rebalance_date - timedelta(days=21), next_rebalance_date + timedelta(days=21), table='benchmark')
        trading_days_encompassing_period = pd.to_datetime(spy_data.index).tolist()

        try:
            if next_rebalance_date not in trading_days_encompassing_period:
                next_rebalance_date = next(trading_day for trading_day in trading_days_encompassing_period if trading_day > next_rebalance_date)
        except StopIteration:
            next_rebalance_date = max(trading_days_encompassing_period)

        try: #try block for lookback_rebalance_date
            if lookback_rebalance_date not in trading_days_encompassing_period:
                lookback_rebalance_date = next(trading_day for trading_day in trading_days_encompassing_period if trading_day < lookback_rebalance_date)
        except StopIteration:
            lookback_rebalance_date = min(trading_days_encompassing_period) # Grab the oldest piece of data

        print('dates for lookback/next ', lookback_rebalance_date, '   ', next_rebalance_date )

        tickers_on_lookback_rebalance_date = db_util.get_tickers_for_date(lookback_rebalance_date.strftime('%Y-%m-%d'))

        tickers_on_next_rebalance_date = db_util.get_tickers_for_date(next_rebalance_date.strftime('%Y-%m-%d'))
        valid_tickers_db = list(set(tickers_on_lookback_rebalance_date) & set(tickers_on_next_rebalance_date))
        ticker_list_sp500, _ = db_util.get_hist_tickers(snap_shot=current_date)
        valid_tickers = list(set(ticker_list_sp500) & set(valid_tickers_db))

        tickers_to_remove = ['TIE', 'MCIC']   #peerhaps errors in db or want to exclude, i.e. GME
        valid_tickers = [ticker for ticker in valid_tickers if ticker not in tickers_to_remove]

        # valid_tickers = valid_tickers[:20]   #for a testing / speeding up backtest


        start = current_date - timedelta(days=self.lookback)
        end = current_date - timedelta(days=1)
        raw_data_dict = self.db.get_prices(valid_tickers, start, end)
        if raw_data_dict is None:
            print('NONE!')
            return None
        raw_data = pd.DataFrame.from_dict(raw_data_dict, orient='index')
        data = raw_data.T  # Transpose the DataFrame to have Ticker as columns and Date as index
        new_weights = self.riskfolio_calc(valid_tickers, current_date, data)
        return new_weights



    def riskfolio_calc(self, assets , current_date, data):
        assets.sort()
        Y = data[assets].pct_change().dropna()
        port = rp.Portfolio(returns=Y)
        method_mu='hist' # Method to estimate expected returns based on historical data.
        method_cov='hist' # Method to estimate covariance matrix based on historical data.
        port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

        port.sht = True # Allows to use Short Weights
        port.uppersht = self.uppersht  #0.3 # Maximum value of sum of short weights in absolute value
        port.upperlng = self.upperlng  #1.3 # Maximum value of sum of positive weights
        port.budget =   self.upperlng - self.uppersht  #self.long_short_budget #1.0 # port.upperlng - port.uppersht

        # Estimate optimal portfolio:
        model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
        rm = 'MV' # Risk measure used, this time will be variance
        obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
        hist = True # Use historical scenarios for risk measures that depend on scenarios
        rf = 0 # Risk free rate
        l = 0 # Risk aversion factor, only useful when obj is 'Utility'
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
        asset_classes = {'Assets': assets, }
        asset_classes = pd.DataFrame(asset_classes)
        asset_classes = asset_classes.sort_values(by=['Assets'])
        constraints = {'Disabled': [False, False, ],
                    'Type': ['All Assets', 'All Assets'],
                    'Set': ['', '' ],
                    'Position': ['', '',],
                    'Sign': ['<=', '>=',],
                    'Weight': [self.max_long_allo, self.max_short_allo],  #0.10 , -0.05
                    'Type Relative': ['', ''],
                    'Relative Set': ['', ''],
                    'Relative': ['', ''],
                    'Factor': ['', '']}
        constraints = pd.DataFrame(constraints)
        A, B = rp.assets_constraints(constraints, asset_classes)
        port.ainequality = A
        port.binequality = B
        model = 'Classic'
        rm = 'MV'
        obj = 'Sharpe'
        rf = 0
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
        w_classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)

        total_leverage = w_classes['weights'].abs().sum()
        long_value = w_classes[w_classes['weights'] > 0]['weights'].sum()
        short_value = w_classes[w_classes['weights'] < 0]['weights'].sum()
        print(f"Total Leverage: {round(total_leverage, 2)}", 
            f"Long Value: {round(long_value, 2)}", 
            f"Short Value: {round(short_value, 2)}")
        weights_dict = w_classes['weights'].to_dict()
        # print(weights_dict)
        return weights_dict









