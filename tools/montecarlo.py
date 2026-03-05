import numpy as np
import random
# from typing import 
import matplotlib.pyplot as plt
from trade import Trade

class MonteCarlo:
    def __init__(self, trades: list[Trade], sample_size_pct: float, with_replacement: bool):
        # only closed trades should be passed in the trades list
        self.trades = trades
        self.sample_size_pct = sample_size_pct
        self.with_replacement = with_replacement

    # Runs a random simulation on the trades once.
    # Returns a value history.
    def run(self) -> np.ndarray:
        # The simulated portfolio start with a relative value of 1
        # New values are added to value_history as this changes
        current_value: float = 1
        value_history = [current_value]

        num_sampled_trades = int(len(self.trades)*self.sample_size_pct/100)
        if self.with_replacement:
            shuffled_trades = random.choices(self.trades, k=num_sampled_trades)
        else:
            shuffled_trades = random.sample(self.trades, num_sampled_trades)

        for trade in shuffled_trades:
            # % change is adjusted for position sizing automatically inside Trade()
            current_value *= (1+trade.pct_change)
            value_history.append(current_value)

        return np.array(value_history)
    
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
            total_percent_change_per_trade += trade.pct_change
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
        print(f"Winrate: {len([i for i in self.trades if i.is_winning])/len(self.trades)} (%)")
        print(f"Average performance: {avg_performance_pct}%")
        for history in value_histories:
            plt.plot(history, alpha=0.1)
        plt.plot(avg_value_history)
        plt.show()
