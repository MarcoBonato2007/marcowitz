from datetime import datetime
import pandas as pd
from typing import Literal, Callable


# Represents a single trade
class Trade:
    def __init__(
        self, 
        ticker: str,
        enter_date: datetime,
        order_type: Literal["Long", "Short"],
        enter_price: float,
        position_size: float, # the % of your cash that goes into this trade (e.g. 0.2)
        take_profit: float, # e.g. input 0.05 for 5%
        stop_loss: float, # e.g. input 0.03 for 3%
        expiration: datetime = None
    ):
        self.ticker = ticker
        self.enter_date = enter_date
        self.order_type = order_type
        self.enter_price = enter_price
        self.position_size = position_size
        self.take_profit = take_profit
        self.stop_loss = stop_loss
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
    def close(self, current_price: float, close_date: datetime):
        self.is_closed = True
        self.set_percents(current_price)
        self.close_price = current_price
        self.close_date = close_date
        if self.pct_change > 0:
            self.is_winning = True
        else:
            self.is_winning = False

    def update(self, current_price: float, date: datetime):
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
    closed_trades: list[Trade] = []
    active_trades: list[Trade] = []

    for i in range(len(df)-1, -1, -30):
        new_trades = [] # new active trades
        cur_df = df.iloc[i:]

        for t in active_trades:
            t.update(
                cur_df.iloc[0]["price"],
                datetime.strptime(cur_df.iloc[0]["date"], '%Y-%m-%d')
            )
            if t.is_closed:
                closed_trades.append(t)
            else:
                new_trades.append(t)

        if (res := judge(cur_df)):
            new_trades.append(res)

        active_trades = new_trades

    return closed_trades
