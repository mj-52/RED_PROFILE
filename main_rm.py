import os
import time
import json
import logging
import concurrent.futures
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
from sklearn.ensemble import RandomForestClassifier
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Load environment variables
load_dotenv()

# Configure logging to be more standard if needed, effectively wrapping global_value.logger
def log(message, level="INFO"):
    global_value.logger(message, level)

class TradingBot:
    def __init__(self):
        # Configuration
        self.OANDA_TOKEN = os.getenv("OANDA_TOKEN")
        self.OANDA_ID = os.getenv("OANDA_ID")
        self.PO_SSID = os.getenv("PO_SSID")
        self.SESSION_START_HOUR = int(os.getenv("SESSION_START_HOUR", 14))
        self.SESSION_END_HOUR = int(os.getenv("SESSION_END_HOUR", 19))
        self.DEMO_MODE = False
                
        # Strategy Settings
        self.MIN_PAYOUT = 80
        self.PERIOD = 300
        self.EXPIRATION = 300
        self.INITIAL_AMOUNT = 1
        self.MARTINGALE_LEVEL = 3
        self.PROB_THRESHOLD = 0.76
        
        self.FEATURE_COLS = ['RSI', 'k_percent', 'r_percent', 'MACD', 'MACD_EMA', 'Price_Rate_Of_Change']
        
        # Initialize APIs
        self.api = PocketOption(self.PO_SSID, self.DEMO_MODE)
        self.oanda_client = oandapyV20.API(access_token=self.OANDA_TOKEN)
        
        # Connect to PocketOption
        self.connect_api()

    def connect_api(self):
        log("🔌 Connecting to PocketOption...", "INFO")
        self.api.connect()
        time.sleep(5)
        if self.api.check_connect():
            log("✅ Connected to PocketOption", "INFO")
        else:
            log("❌ Connection failed", "ERROR")

    def is_trade_time(self):
        """
        Checks if current time is within the allowed trading session (default 14:00 - 19:00 EAT).
        Returns: (bool, str) -> (is_allowed, reason/status)
        """
        # EAT is UTC+3
        eat_offset = timedelta(hours=3)
        now_utc = datetime.now(timezone.utc)
        now_eat = now_utc + eat_offset
        
        current_hour = now_eat.hour
        
        if self.SESSION_START_HOUR <= current_hour < self.SESSION_END_HOUR:
            return True, f"Trading active ({now_eat.strftime('%H:%M')} EAT)"
        else:
            return False, f"Trading disabled ({now_eat.strftime('%H:%M')} EAT)"

    def get_oanda_candles(self, pair, granularity="M5", count=500):
        try:
            oanda_pair = pair[:3] + "_" + pair[3:] if "_" not in pair else pair
            params = {"granularity": granularity, "count": count}
            r = instruments.InstrumentsCandles(instrument=oanda_pair, params=params)
            self.oanda_client.request(r)
            candles = r.response['candles']
            
            data = []
            for c in candles:
                data.append({
                    'time': c['time'],
                    'open': float(c['mid']['o']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']),
                    'close': float(c['mid']['c']),
                })
                
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            return pair, df
        except Exception as e:
            # log(f"[ERROR]: OANDA fetch failed for {pair} - {str(e)}", "ERROR")
            return pair, None

    def get_payouts(self):
        try:
            d = json.loads(global_value.PayoutData)
            valid_pairs = {}
            for pair in d:
                name = pair[1]
                payout = pair[5]
                asset_type = pair[3]
                is_active = pair[14]

                if not name.endswith("_otc") and asset_type == "currency" and is_active:
                    if payout >= self.MIN_PAYOUT:
                        valid_pairs[name] = {'payout': payout, 'type': asset_type}
            
            # Update global pairs if needed or return them
            global_value.pairs = valid_pairs # syncing with legacy global usage if any
            return valid_pairs
        except Exception as e:
            log(f"[ERROR]: Failed to parse payout data - {str(e)}", "ERROR")
            return {}

    def prepare_data(self, df):
        df = df.copy()
        df.rename(columns={'time': 'timestamp'}, inplace=True)
        # df.sort_values(by='timestamp', inplace=True) # Already sorted usually

        # Technical Indicators
        df['RSI'] = ta.rsi(df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['k_percent'] = stoch['STOCHk_14_3_3']
        df['r_percent'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_EMA'] = macd['MACDs_12_26_9']
        df['Price_Rate_Of_Change'] = ta.roc(df['close'], length=9)
        supert = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['SUPERT_10_3.0'] = supert['SUPERT_10_3.0']
        df['SUPERTd_10_3.0'] = supert['SUPERTd_10_3.0']

        df['Prediction'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def add_pivots(self, df):

        n_left = 10
        n_right = 10
        window = n_left + n_right + 1

        roll_high = df['high'].rolling(window=window, center=True).max()
        roll_low = df['low'].rolling(window=window, center=True).min()

        df['is_high'] = (df['high'] == roll_high)
        df['is_low'] = (df['low'] == roll_low)
        
        # Map to original integer code: 1=low, 2=high, 3=both(rare)
        def get_pivot_val(row):
            if row['is_high'] and row['is_low']: return 3
            if row['is_high']: return 2
            if row['is_low']: return 1
            return 0
            
        df['pivot'] = 0
        df.loc[df['is_low'], 'pivot'] = 1
        df.loc[df['is_high'], 'pivot'] = 2
        df.loc[df['is_high'] & df['is_low'], 'pivot'] = 3
        
        # Clean up last n_right rows where logic is invalid (handled by rolling NaN usually but good to be explicit if needed)
        # Pandas rolling center=True produces NaNs at ends.
        df['pivot'] = df['pivot'].fillna(0).astype(int)
        
        return df

    def train_and_predict(self, df):
        # Prepare Data
        df = self.prepare_data(df)
        
        # Model Training (Retraining every time is expensive but sticking to original logic)
        X_train = df[self.FEATURE_COLS].iloc[:-1]
        y_train = df['Prediction'].iloc[:-1]

        model = RandomForestClassifier(n_estimators=100, oob_score=True, criterion="gini", random_state=0, n_jobs=-1) # Parallel jobs
        model.fit(X_train, y_train)

        X_test = df[self.FEATURE_COLS].iloc[[-1]]
        proba = model.predict_proba(X_test)
        call_conf = proba[0][1]
        put_conf = 1 - call_conf

        # Strategy Logic
        latest = df.iloc[-1]
        latest_dir = latest['SUPERTd_10_3.0']
        current_trend = latest['SUPERT_10_3.0']
        past_trend = df.iloc[-3]['SUPERT_10_3.0']
        rsi = latest['RSI']
        current_price = latest['close']

        # Pivots
        df = self.add_pivots(df)
        
        pivot_highs = df[df['pivot'] == 2]
        latest_pivot_high = pivot_highs.iloc[-1]['high'] if not pivot_highs.empty else None
        
        pivot_lows = df[df['pivot'] == 1]
        latest_pivot_low = pivot_lows.iloc[-1]['low'] if not pivot_lows.empty else None

        # Filters
        if rsi > 70 or rsi < 30:
            log(f"⏭️ [RSI] Overbought/Oversold ({rsi:.2f})", "INFO")
            return None

        if current_trend == past_trend:
             log(f"⏭️ [TREND] Flat trend", "INFO")
             return None

        # Decision
        decision = None
        confidence = 0.0
        emoji = ""

        if call_conf > self.PROB_THRESHOLD:
            if latest_dir == 1 and latest_pivot_high and current_price < latest_pivot_high:
                decision = "call"
                emoji = "🟢"
                confidence = call_conf
            else:
                 log(f"⏭️ [CALL] Filtered: Trend/Pivot mismatch ({call_conf:.1%})", "INFO")
        elif put_conf > self.PROB_THRESHOLD:
            if latest_dir == -1 and latest_pivot_low and current_price > latest_pivot_low:
                decision = "put"
                emoji = "🔴"
                confidence = put_conf
            else:
                 log(f"⏭️ [PUT] Filtered: Trend/Pivot mismatch ({put_conf:.1%})", "INFO")
        
        if decision:
            log(f"{emoji} PREDICTION: {decision.upper()} | Conf: {confidence:.2%}", "INFO")
            return decision

        return None

    def execute_trade(self, pair, action):
        log(f"🚀 Executing {action.upper()} on {pair}", "INFO")
        
        amount = self.INITIAL_AMOUNT
        for level in range(1, self.MARTINGALE_LEVEL + 1):
            result = self.api.buy(amount=amount, active=pair, action=action, expirations=self.EXPIRATION)
            if not result or not result[1]:
                log("❗ Trade Request Failed. Reconnecting...", "ERROR")
                self.api.disconnect()
                time.sleep(2)
                self.connect_api()
                return

            trade_id = result[1]
            time.sleep(self.EXPIRATION)
            # Check win returns a tuple (bool, status) e.g. (True, 'win')
            result = self.api.check_win(trade_id)
            win_status = result[1] if isinstance(result, tuple) and len(result) > 1 else result
            
            if win_status == 'win':
                log(f"💰 WIN (Level {level})", "INFO")
                return
            elif win_status == 'loose':
                log(f"💸 LOSS (Level {level})", "INFO")
                amount *= 2 # Martingale
            else:
                log(f"❓ Unknown Result: {result}", "WARNING")
                return

    def wait_for_data(self):
        # Wait until 20 seconds before next candle
        # Candle starts at minute % 5 == 0.
        while True:
            now = datetime.now(timezone.utc)
            # Find next candle start time
            current_timestamp = now.timestamp()
            next_candle_start = ((current_timestamp // self.PERIOD) + 1) * self.PERIOD
            
            seconds_remaining = next_candle_start - current_timestamp
            
            if seconds_remaining <= 20:
                break
            
            sleep_time = min(seconds_remaining - 20, 1) # Wake up mostly at 20s mark, but sleep in chunks
            time.sleep(max(0.1, sleep_time))

    def wait_for_candle(self):
         while True:
            now = datetime.now(timezone.utc)
            if now.second == 0 and now.minute % (self.PERIOD // 60) == 0:
                break
            time.sleep(0.1)

    def run(self):
        log("🤖 Bot Started", "INFO")
        while True:
            is_active, status = self.is_trade_time()
            if not is_active:
                log(f"💤 {status}", "INFO")
                time.sleep(60)
                continue
            
            log(f"🔄 Cycle Start | {status}", "INFO")
            
            pairs_map = self.get_payouts()
            if not pairs_map:
                log("❗ No valid pairs found.", "ERROR")
                time.sleep(10)
                continue

            self.wait_for_data()
            log("⏳ Analyzing Markets...", "INFO")
            
            # Parallel Data Fetching
            selected_pair = None
            selected_action = None
            
            # Use ThreadPool to fetch OANDA data in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self.get_oanda_candles, pair): pair for pair in pairs_map.keys()}
                
                for future in concurrent.futures.as_completed(futures):
                    pair, df = future.result()
                    
                    if df is None or len(df) < 50:
                        continue
                        
                    decision = self.train_and_predict(df)
                    if decision:
                        selected_pair = pair
                        selected_action = decision
                        # We cancel pending futures? 
                        # To stop processing other pairs if we found a trade?
                        # The original logic stopped at 'break' (first match).
                        # With parallel, we might find multiple. Let's take the first one that finishes (Race).
                        
                        # Stop others (optional, or just ignore their results)
                        executor.shutdown(wait=False, cancel_futures=True)
                        break 
            
            self.wait_for_candle()
            
            if selected_pair and selected_action:
                self.execute_trade(selected_pair, selected_action)
            else:
                 log("⛔ No Setup Found", "INFO")
                 
            time.sleep(2)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()


