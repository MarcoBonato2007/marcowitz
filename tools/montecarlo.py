import numpy as np
import random
import matplotlib.pyplot as plt
from tools.trade import Trade

class MonteCarlo:
    # Pass sample_size_pct as a percentage between 0 and 1
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

        num_sampled_trades = int(len(self.trades)*self.sample_size_pct)
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
        running_max = np.maximum.accumulate(value_history)
        max_drawdown_percent = ((running_max-value_history)/running_max).max()
        performance_pct = value_history[-1]
        return performance_pct, max_drawdown_percent
    
    def run_many(self, num_runs: int) -> np.ndarray:
        # Do a portfolio simulation on <num_runs> portfolios
        # Return a 2D list of all their price values over time
        value_histories: np.ndarray = np.array(self.run())
        for _ in range(num_runs-1): # -1 since we already did it once in initializing value histories
            value_histories = np.vstack((value_histories, self.run()))
        return value_histories

    def get_avg_stats(self, value_histories: np.ndarray) -> tuple[float, float, float, float, float, np.ndarray]:        
        # Calculate avg pct change per trade
        total_pct_change = 0
        for trade in self.trades:
            total_pct_change += trade.pct_change
        avg_pct_change = total_pct_change / len(self.trades)

        # Calculate the max drawdown and least performance (across all runs)
        max_drawdown_pct = 0
        least_performance_pct = np.inf
        for history in value_histories:
            performance_pct, drawdown_pct = self.get_stats(history)
            least_performance_pct = min(least_performance_pct, performance_pct)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        # Calculate average performance
        avg_value_history: np.ndarray = value_histories.sum(axis=0) / value_histories.shape[0]
        avg_performance_pct = avg_value_history[-1]

        return avg_performance_pct, max_drawdown_pct, least_performance_pct, avg_pct_change, avg_value_history

    def show(self, num_runs: int):
        # shows important stats and plots a graph
        value_histories: np.ndarray = self.run_many(num_runs)
        avg_performance_pct, max_drawdown_percent, least_performance_pct, avg_percent_change_per_trade, avg_value_history = self.get_avg_stats(value_histories)
        print(f"Max drawdown: -{max_drawdown_percent*100}%")
        print(f"Worst performance: {least_performance_pct*100}%")
        print(f"Avg. gain per trade: {avg_percent_change_per_trade*100}%")
        print(f"Winrate: {100*len([i for i in self.trades if i.is_winning])/len(self.trades)}%")
        print(f"Average performance: {avg_performance_pct*100}%")
        for history in value_histories:
            plt.plot(history, alpha=0.1)
        plt.plot(avg_value_history)
        plt.show()
