import pandas as pd
from typing import Literal, Callable

# Represents a single trade
class Trade:
    def __init__(
        self, 
        ticker: str,
        enter_date: str,
        order_type: Literal["Long", "Short"],
        enter_price: float,
        position_size: float, # the % of your cash that goes into this trade (e.g. 0.2)
        take_profit: float, # e.g. input 0.05 for 5%
        stop_loss: float, # e.g. input 0.03 for 3%
        expiration: str = None
    ):
        self.ticker = ticker
        self.enter_date = enter_date
        self.order_type = order_type
        self.enter_price = enter_price
        self.position_size = position_size
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.take_profit_price = self.enter_price*(1+take_profit)
        self.stop_loss_price = self.enter_price*(1-stop_loss)
        self.raw_pct_change: float = 0 # e.g. (most recent price-enter price) / enter price (if long)
        self.pct_change: float = 0 # adjusted for position size
        self.expiration = expiration
        self.close_price = None
        self.close_date = None
        self.is_closed = False
        self.is_winning = None

    def set_percents(self, current_price: float) -> None:
        self.raw_pct_change = (current_price-self.enter_price)/self.enter_price
        if (self.order_type == "Short"):
            self.raw_pct_change = -self.raw_pct_change
        self.pct_change = self.raw_pct_change*self.position_size

    # Close this trade, rendering it unresponsive to update() calls
    def close(self, current_price: float, close_date: str):
        self.is_closed = True
        self.set_percents(current_price)
        self.close_price = current_price
        self.close_date = close_date
        if self.pct_change > 0:
            self.is_winning = True
        else:
            self.is_winning = False

    # Pass in a standard dataframe
    # Finds where stop loss / take profit is hit
    def simulate(self, df):
        df = df[df["date"] >= self.enter_date]
        tp = df[(df["low"] <= self.take_profit_price) & (df["high"] >= self.take_profit_price)]
        sl = df[(df["low"] <= self.stop_loss_price) & (df["high"] >= self.stop_loss_price)]
        sl_date = None
        tp_date = None
        if len(sl) >= 1: sl_date = sl.iat[-1, 0]
        if len(tp) >= 1: tp_date = tp.iat[-1, 0]
        if (sl_date == None and tp_date == None): pass # no sl or tp ever hit
        elif (sl_date != None and (tp_date == None or sl_date < tp_date)):
            self.close(self.stop_loss_price, sl_date)
        else:
            self.close(self.take_profit_price, tp_date)

    def update(self, current_price: float, date: str):
        if self.is_closed:
            return
        
        self.set_percents(current_price)
        if (self.raw_pct_change > self.take_profit) or (self.raw_pct_change < -self.stop_loss):
            self.close(current_price, date)
        elif self.expiration != None and date > self.expiration:
            self.close(current_price, date)

# Judge is a function that takes in a dataframe, and returns a trade to make (if any)
# The dataframe must have:
    # A date column, with latest date at the top, formatted "yyyy-mm-dd"
    # A price column
# get_trades will simulate starting from the inception of the stock, and jumping by months,
# each month checking if a trade should be made. Trades are kept track of.
# Only trades that are closed are returned.
def get_trades(df, judge: Callable[[pd.DataFrame], Trade | None]):
    all_trades: list[Trade] = []

    for i in range(len(df)-1, -1, -30):
        cur_df = df.iloc[i:]
        if (res := judge(cur_df)):
            all_trades.append(res)

    out = []
    for t in all_trades:
        t.simulate(df)
        if t.is_closed:
            out.append(t)

    return out
