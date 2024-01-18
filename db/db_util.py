import sqlite3
import pandas as pd
import datetime


class Database:
    def __init__(self, db_name='db/stocks.db'):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)

    def close(self):
        self.conn.close()

    def get_price(self, symbol, start_date, end_date=None, table='sp500_data', today_only=False):
        if today_only:
            end_date = start_date + datetime.timedelta(days=1) 
        query = f"""
            SELECT `Date`, `Adj Close`
            FROM {table}
            WHERE `Date` >= '{start_date}' AND `Date` < '{end_date}' AND `Ticker` = '{symbol}'
        """
        try:
            prices = pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"Failed to get prices. Reason: {e}")
            return None

        if prices.empty:  # If the returned DataFrame is empty
            return None
        else:
            return prices.set_index('Date')  # Return the DataFrame with the Date column as index




    def get_prices(self, symbols, start_date, end_date=None, table='sp500_data', today_only=False):
        if today_only:
            end_date = start_date + datetime.timedelta(days=1) 

        symbols_str = ', '.join([f"'{symbol}'" for symbol in symbols])
        query = f"""
            SELECT Ticker, `Date`, `Adj Close`
            FROM {table}
            WHERE `Date` >= '{start_date}' AND `Date` < '{end_date}' AND Ticker IN ({symbols_str})
        """
        try:
            prices = pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"Failed to get prices. Reason: {e}")
            return None

        if prices.empty:  # If the returned DataFrame is empty
            return None
        else:
            # Pivot the DataFrame to have Ticker as index, Date as columns, and Adj Close as values
            prices_pivot = prices.pivot(index='Ticker', columns='Date', values='Adj Close')
            # Convert the pivoted DataFrame to a dictionary of dictionaries
            return prices_pivot.to_dict(orient='index')



def get_tickers_for_date(date, table_name='sp500_data', db_name='db/stocks.db'):
    """
    Get list of tickers that have data for the given date
    :param date: str, date in the format 'YYYY-MM-DD'
    :param table_name: str, name of the table in the database to pull data from
    :param db_name: str, path to the SQLite database file
    :return: list of tickers
    """
    # Open connection to SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    date = datetime.datetime.strptime(date, "%Y-%m-%d") 
    # print(date)

    # Execute query to get distinct tickers that have data on the given date
    cursor.execute(f"SELECT DISTINCT ticker FROM {table_name} WHERE Date = ?", [date])
    tickers = [row[0] for row in cursor.fetchall()]

    # Close connection to SQLite database
    conn.close()
    tickers = fix_ticker_dash(tickers)
    return tickers

def fix_ticker_dash(tickers):
    # Needs to work if its a list of tickers or a single ticker
    # if list return list, if a single ticker return like form
    if isinstance(tickers, list): # checks if 'tickers' is a list
        clean_tickers = [x.replace('.','-') for x in tickers]
    elif isinstance(tickers, str): # checks if 'tickers' is a string
        clean_tickers = tickers.replace('.', '-')
    else:
        raise ValueError('Input should be either a string or list of strings')

    return clean_tickers

def get_hist_tickers(snap_shot = '2023-10-23'):
    filename = 'db/sp500_hist.csv'
    df = pd.read_csv(filename, index_col='date', parse_dates=['date'])
    df.sort_index(inplace=True)  # Ensure the DataFrame is sorted by date

    df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))

    # Get all the rows up to and including the 'snap_shot' date
    df2 = df[df.index <= pd.to_datetime(snap_shot)]
    if df2.empty:
        print("No data for and before this date.")
        return None, None

    # Get the last row (most recent date not after snap_shot)
    last_row = df2.iloc[-1]
    past_tickers = last_row['tickers']
    date = last_row.name.strftime('%Y-%m-%d')  # Format date as a string

    return past_tickers, date











