import pandas as pd
import numpy as np


class TradingEnvironment:
    def __init__(self, data, initial_balance=1000000, trading_fee=0.001, max_steps=None):
        self.data = data.reset_index()
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee  # Comisión por transacción
        self.max_steps = max_steps if max_steps else len(data) - 1
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.done = False
        self.trade_history = []
        self.positions = []
        self.port_val_history = [initial_balance]
        self.winloss_history = []

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.done = False
        self.trade_history = []
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.data.loc[self.current_step, 'Close'],
            self.data.loc[self.current_step, 'SMA_50'],
            self.data.loc[self.current_step, 'SMA_200'],
            self.data.loc[self.current_step, 'RSI_14'],
            self.data.loc[self.current_step, 'MACD'],
            self.shares_held,
            self.balance
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise Exception("El episodio ha terminado, reinicia el entorno.")

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        current_price = self.data.loc[self.current_step, 'Close']
        reward = 0
        fee = 0

        if action == 1:  # Comprar
            if self.balance >= current_price:
                fee = current_price * self.trading_fee
                self.shares_held += 1
                self.balance -= (current_price + fee)
                self.trade_history.append((self.current_step, 'BUY', current_price, fee))
                self.positions.append(current_price)
                reward = 0.01

        elif action == 2:  # Vender
            if self.shares_held > 0:
                fee = current_price * self.trading_fee
                self.shares_held -= 1
                self.balance += (current_price - fee)
                self.trade_history.append((self.current_step, 'SELL', current_price, fee))
                original_price = self.positions.pop(0)
                reward = (current_price - original_price) / original_price
                self.winloss_history.append(reward>0)

        self.port_val_history.append(self.balance + (self.shares_held * current_price))
        self.total_profit = self.balance + (self.shares_held * current_price) - self.initial_balance
        return self._get_observation(), reward, self.done

    def render(self):
        print(
            f'Step: {self.current_step}, Balance: {self.balance:.2f}, Shares Held: {self.shares_held}, Total Profit: {self.total_profit:.2f}')

    def get_trade_history(self):
        return pd.DataFrame(self.trade_history, columns=['Step', 'Action', 'Price', 'Fee'])

#Metricas
#Sharpe's Ratio, Sortino, Calmar Ratio, Max Drawdown, WinLoss
