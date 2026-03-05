import numpy as np
import random
import datetime
from typing import *
import matplotlib.pyplot as plt

# Judge takes a df as input
# With latest date first
# Returns a trade if it wants to trade, otherwise returns None
# Jumps by a month each time
def get_trades(df, judge):
    # Must have latest date as first element
    closed_trades: list[Trade] = []
    active_trades: list[Trade] = []

    for i in range(len(df)-1, -1, -30):
        new_trades = [] # new active trades
        cur_df = df.iloc[i:]

        for t in active_trades:
            t.update(
                cur_df.iloc[0]["avg"],
                datetime.datetime(*map(int, cur_df.iloc[0]["date"].split('-')))
            )
            if t.is_closed:
                closed_trades.append(t)
            else:
                new_trades.append(t)

        if (res := judge(cur_df)):
            new_trades.append(res)

        active_trades = new_trades

    return closed_trades

# Represents a single trade.
class Trade:
    def __init__(
        self, 
        symbol: str,
        enter_date: datetime.datetime,
        order_type: Literal["Long", "Short"],
        enter_price: float, # price at the point where you enter
        position_size_percent: float, # what % of your cash goes into this trade?
        take_profit_percent: float, # e.g. input 5 for 5%
        stop_loss_percent: float, # e.g. input 3 for 3%
        expiration_date: datetime.datetime = None
    ):
        self.symbol = symbol
        self.enter_date = enter_date
        self.order_type = order_type
        self.enter_price = enter_price
        self.position_size_percent = position_size_percent
        self.take_profit_percent = take_profit_percent
        self.stop_loss_percent = stop_loss_percent
        self.percent_change: float = 0
        self.expiration_date = expiration_date
        self.close_price = None
        self.close_date = None
        self.is_closed = False
        self.is_winning = None

    def get_percent_change(self, current_price: float) -> tuple[float, float]:
        raw_percent_change = 0
        if self.order_type == "Long":
            raw_percent_change = (current_price-self.enter_price)/self.enter_price*100
        else:
            raw_percent_change = (self.enter_price-current_price)/self.enter_price*100
        position_size_adjusted_pct_change = raw_percent_change * self.position_size_percent/100
        return raw_percent_change, position_size_adjusted_pct_change

    def close(self, current_price, close_date: datetime.datetime):
        self.is_closed = True
        _, self.percent_change = self.get_percent_change(current_price)
        self.close_price = current_price
        self.close_date = close_date
        if self.percent_change > 0:
            self.is_winning = True
        else:
            self.is_winning = False

    def update(self, current_price: float, date: datetime.datetime):
        if self.is_closed:
            return
        
        raw_pct_change, self.percent_change = self.get_percent_change(current_price)
        if (raw_pct_change > self.take_profit_percent) or (raw_pct_change < -self.stop_loss_percent):
            self.close(current_price, date)
        elif self.expiration_date != None and date > self.expiration_date:
            self.close(current_price, date)

class MonteCarlo:
    def __init__(self, trades: list[Trade], sample_size_pct: float, with_replacement: bool):
        # only closed trades should be passed in the trades list
        self.trades = trades
        self.sample_size_pct = sample_size_pct
        self.with_replacement = with_replacement

    def run(self) -> np.ndarray:
        # The simulated portfolio start with a relative value of 1
        # New values are added to value_history as this changes
        current_value: float = 1
        value_history: np.ndarray = np.array([current_value])

        num_sampled_trades = int(len(self.trades)*self.sample_size_pct/100)
        if self.with_replacement:
            shuffled_trades = random.choices(self.trades, k=num_sampled_trades)
        else:
            shuffled_trades = random.sample(self.trades, num_sampled_trades)

        for trade in shuffled_trades:
            # % change is adjusted for position sizing automatically inside Trade()
            current_value *= (1+trade.percent_change/100)
            value_history = np.append(value_history, current_value)

        return value_history
    
    def get_stats(self, value_history: np.ndarray) -> tuple[float, float, float]:
        # returns max drawdown and std deviation
        running_max = np.maximum.accumulate(value_history)
        max_drawdown_percent = ((running_max-value_history)/running_max).max()*100
        std_deviation = value_history.std()
        performance_pct = value_history[-1]*100
        return performance_pct, max_drawdown_percent, std_deviation
    
    def run_many(self, num_runs: int) -> np.ndarray:
        # Do a portfolio simulation on <num_times> portfolios
        # Return a 2D list of all their price values over time
        value_histories: np.ndarray = np.array(self.run())
        for _ in range(num_runs-1): # -1 since we already did it once in initializing value histories
            value_histories = np.vstack((value_histories, self.run()))

        return value_histories

    def get_avg_stats(self, value_histories: np.ndarray) -> tuple[float, float, float, float, float, np.ndarray]:
        # returns max drawdown, avg std.dev, avg value history
        avg_value_history: np.ndarray = value_histories.sum(axis=0) / value_histories.shape[0]
        
        total_percent_change_per_trade = 0
        for trade in self.trades:
            total_percent_change_per_trade += trade.percent_change
        avg_percent_change_per_trade = total_percent_change_per_trade / len(self.trades)

        running_max_drawdown_percent = 0
        total_std_dev = 0
        least_performance_pct = 0
        for history in value_histories:
            performance_pct, max_drawdown_percent, std_dev = self.get_stats(history)
            least_performance_pct = min(performance_pct, least_performance_pct)
            running_max_drawdown_percent = max(max_drawdown_percent, running_max_drawdown_percent)
            total_std_dev += std_dev
        avg_std_dev = total_std_dev / len(value_histories)

        avg_performance_pct = avg_value_history[-1]
        return avg_performance_pct, running_max_drawdown_percent, least_performance_pct, avg_std_dev, avg_percent_change_per_trade, avg_value_history

    def show(self, num_runs: int):
        # shows important stats and plots a graph
        value_histories: np.ndarray = self.run_many(num_runs)
        avg_performance_pct, max_drawdown_percent, least_performance_pct, avg_std_dev, avg_percent_change_per_trade, avg_value_history = self.get_avg_stats(value_histories)
        print(f"Max drawdown: -{max_drawdown_percent}%")
        print(f"Worst performance: {least_performance_pct}%")
        print(f"Avg std dev: {avg_std_dev}")
        print(f"Avg. % gain per trade: {avg_percent_change_per_trade}")
        print(f"Winrate: {len([i for i in self.trades if i.is_winning])/len(self.trades)*100}%")
        print(f"Average performance: {avg_performance_pct}%")
        for history in value_histories:
            plt.plot(history, alpha=0.1)
        plt.plot(avg_value_history)
        plt.show()
