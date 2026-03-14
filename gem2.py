import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_and_clean_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        
        # Ensure a continuous time series
        full_date_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')
        self.df = self.df.reindex(full_date_range)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        
        # Correct price glitches
        for col in ['Open', 'High', 'Low', 'Close']:
            self.df[col] = self.df[col].apply(lambda x: np.nan if x <= 0 else x)
        self.df.interpolate(method='linear', inplace=True)
        
        return self.df

class StateMachineStrategy:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def generate_signals(self) -> pd.DataFrame:
      
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        
        self.df['TR'] = np.max(ranges, axis=1)
        self.df['ATR'] = self.df['TR'].rolling(14).mean()
        self.df['ATR_MA30'] = self.df['ATR'].rolling(30).mean()
        
        
        self.df['Is_Volatile'] = self.df['ATR'] > (1.2 * self.df['ATR_MA30'])
        
      
        self.df['EMA_10'] = self.df['Close'].ewm(span=10, adjust=False).mean()
        self.df['EMA_30'] = self.df['Close'].ewm(span=30, adjust=False).mean()
        
        self.df['EMA_30_v'] = self.df['Close'].ewm(span=30, adjust=False).mean()
        self.df['EMA_100_v'] = self.df['Close'].ewm(span=100, adjust=False).mean()
        
        self.df['Signal'] = 0
        self.df['Stop_Loss_Mult'] = 0.0  
        

        calm_buy = (~self.df['Is_Volatile']) & (self.df['EMA_10'] > self.df['EMA_30']) & (self.df['EMA_10'].shift(1) <= self.df['EMA_30'].shift(1))
        calm_sell = (~self.df['Is_Volatile']) & (self.df['EMA_10'] < self.df['EMA_30']) & (self.df['EMA_10'].shift(1) >= self.df['EMA_30'].shift(1))
        

        vol_buy = (self.df['Is_Volatile']) & (self.df['EMA_30_v'] > self.df['EMA_100_v']) & (self.df['EMA_30_v'].shift(1) <= self.df['EMA_100_v'].shift(1))
        vol_sell = (self.df['Is_Volatile']) & (self.df['EMA_30_v'] < self.df['EMA_100_v']) & (self.df['EMA_30_v'].shift(1) >= self.df['EMA_100_v'].shift(1))
        
 
        self.df.loc[calm_buy, ['Signal', 'Stop_Loss_Mult']] = [1, 1.0] 
        self.df.loc[vol_buy, ['Signal', 'Stop_Loss_Mult']] = [1, 2.0] 
        
        self.df.loc[calm_sell | vol_sell, 'Signal'] = -1
        
        return self.df

class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital=10000.0, transaction_cost=0.0015):
        self.df = df
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0 
        self.transaction_cost = transaction_cost 
        
        self.trades = []
        self.current_trade = None
        self.equity_curve = []

    def run(self):
        for row in self.df.itertuples():
            price = row.Close
            signal = row.Signal
            atr = row.ATR
            date = row.Index
            sl_mult = getattr(row, 'Stop_Loss_Mult', 0)
            
           
            if self.position > 0 and self.current_trade:
                stop_loss_price = self.current_trade['entry_price'] - (self.current_trade['sl_mult'] * self.current_trade['entry_atr'])
                
               
                if row.Low <= stop_loss_price:
                    exit_price = min(row.Open, stop_loss_price)
                    self.sell(exit_price, date, 'Stop Loss')
                    
            
            if signal == 1 and self.position == 0:
                self.buy(price, date, atr, sl_mult)
            elif signal == -1 and self.position > 0:
                self.sell(price, date, 'Signal Exit')
                
            
            current_equity = self.capital + (self.position * price)
            self.equity_curve.append(current_equity)
            
       
        if self.position > 0:
            self.sell(self.df.iloc[-1]['Close'], self.df.index[-1], 'End of Period')
            
        self.df['Equity'] = self.equity_curve
        return self.calculate_metrics()
        
    def buy(self, price, date, atr, sl_mult):
        cost = self.capital * self.transaction_cost
        investment = self.capital - cost
        self.position = investment / price
        
        self.current_trade = {
            'entry_date': date,
            'entry_price': price,
            'entry_atr': atr,
            'sl_mult': sl_mult, 
            'capital_invested': self.capital
        }
        self.capital = 0
        
    def sell(self, price, date, reason):
        if not self.current_trade: return
        
        gross_value = self.position * price
        cost = gross_value * self.transaction_cost
        net_value = gross_value - cost
        pnl = net_value - self.current_trade['capital_invested']
        
        self.trades.append({
            'entry_date': self.current_trade['entry_date'],
            'exit_date': date,
            'pnl': pnl,
            'reason': reason
        })
        
        self.capital = net_value
        self.position = 0
        self.current_trade = None

    def calculate_metrics(self):
        trades_df = pd.DataFrame(self.trades)
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if not trades_df.empty else 0
        gross_loss = trades_df[trades_df['pnl'] < 0]['pnl'].sum() if not trades_df.empty else 0
        net_profit = gross_profit + gross_loss
        
        total_closed_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0]) if not trades_df.empty else 0
        win_rate = (winning_trades / total_closed_trades) * 100 if total_closed_trades > 0 else 0
        
        start_price = self.df['Close'].iloc[0]
        end_price = self.df['Close'].iloc[-1]
        bnh_return = ((end_price - start_price) / start_price) * 100
        strategy_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        self.df['Peak'] = self.df['Equity'].cummax()
        self.df['Drawdown'] = (self.df['Equity'] - self.df['Peak']) / self.df['Peak']
        max_drawdown = self.df['Drawdown'].min() * 100
        
        self.df['Daily_Return'] = self.df['Equity'].pct_change()
        mean_return = self.df['Daily_Return'].mean()
        std_return = self.df['Daily_Return'].std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(365) if std_return != 0 else 0
        
        downside_returns = self.df[self.df['Daily_Return'] < 0]['Daily_Return']
        downside_std = downside_returns.std()
        sortino_ratio = (mean_return / downside_std) * np.sqrt(365) if downside_std != 0 else 0
        
        return {
            'Gross Profit (USDT)': round(gross_profit, 2),
            'Gross Loss (USDT)': round(gross_loss, 2),
            'Net Profit (USDT)': round(net_profit, 2),
            'Strategy Return (%)': round(strategy_return, 2),
            'Buy & Hold Return (%)': round(bnh_return, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),   
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Sortino Ratio': round(sortino_ratio, 2),
            'Total Closed Trades': total_closed_trades,
            'Win Rate (%)': round(win_rate, 2)
        }

if __name__ == "__main__":
    processor = DataProcessor('Beat_The_Market_Dataset (1).csv')
    df_clean = processor.load_and_clean_data()
    
    strategy = StateMachineStrategy(df_clean)
    df_signals = strategy.generate_signals()
    
    backtester = Backtester(df_signals, initial_capital=10000.0, transaction_cost=0.0015)
    final_metrics = backtester.run()
    
    print("\n" + "="*45)
    print("  FINAL HYPER-OPTIMIZED BACKTEST RESULTS ")
    print("="*45)
    for metric, value in final_metrics.items():
        if metric == 'Strategy Return (%)':
            print(f" {metric}: {value} ")
        elif metric == 'Max Drawdown (%)':
            print(f"  {metric}: {value} (Safe: <30%)")
        else:
            print(f"{metric}: {value}")
    print("="*45)