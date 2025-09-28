import os
import time
import json
import pandas as pd
import pandas_ta as ta 
from datetime import datetime, timezone
from dotenv import load_dotenv
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
from sklearn.ensemble import RandomForestClassifier
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Load environment variables
load_dotenv()

# === Credentials ===
ACCESS_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ID")
ssid = os.getenv("PO_SSID")
demo = True

# Bot Settings
min_payout = 80
period = 300
expiration = 300
INITIAL_AMOUNT = 1
MARTINGALE_LEVEL = 3
PROB_THRESHOLD = 0.60

api = PocketOption(ssid, demo)
api.connect()
time.sleep(5)

FEATURE_COLS = ['RSI', 'k_percent', 'r_percent', 'MACD', 'MACD_EMA', 'Price_Rate_Of_Change']

def get_oanda_candles(pair, granularity="M5", count=500):
    try:
        client = oandapyV20.API(access_token=ACCESS_TOKEN)
        params = {"granularity": granularity, "count": count}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        candles = r.response['candles']
        df = pd.DataFrame([{
            'time': c['time'],
            'open': float(c['mid']['o']),
            'high': float(c['mid']['h']),
            'low': float(c['mid']['l']),
            'close': float(c['mid']['c']),
        } for c in candles])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        global_value.logger(f"[ERROR]: OANDA candle fetch failed for {pair} ", "ERROR")
        return None

def get_payout():
    try:
        d = json.loads(global_value.PayoutData)
        for pair in d:
            name = pair[1]
            payout = pair[5]
            asset_type = pair[3]
            is_active = pair[14]

            if not name.endswith("_otc") and asset_type == "currency" and is_active:
                if payout >= min_payout:
                    global_value.pairs[name] = {'payout': payout, 'type': asset_type}
                elif name in global_value.pairs:
                    del global_value.pairs[name]
        return True
    except Exception as e:
        global_value.logger(f"[ERROR]: Failed to parse payout data - {str(e)}", "ERROR")
        return False

def prepare_data(df):
    df = df[['time', 'open', 'high', 'low', 'close']]
    df.rename(columns={'time': 'timestamp'}, inplace=True)
    df.sort_values(by='timestamp', inplace=True)

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
    df.reset_index(drop=True, inplace=True)  # <-- Add this line
    return df

def pivotid(df1, l, n1, n2):
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    pividlow = 1
    pividhigh = 1
    for i in range(l - n1, l + n2 + 1):
        if df1.low[l] > df1.low[i]:
            pividlow = 0
        if df1.high[l] < df1.high[i]:
            pividhigh = 0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0

def train_and_predict(df):
    X_train = df[FEATURE_COLS].iloc[:-1]
    y_train = df['Prediction'].iloc[:-1]

    # global_value.logger("📊 Latest data preview:\n" + str(df.shape), "INFO")
    model = RandomForestClassifier(n_estimators=100, oob_score=True, criterion="gini", random_state=0)
    model.fit(X_train, y_train)

    X_test = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(X_test)
    call_conf = proba[0][1]
    put_conf = 1 - call_conf

    latest_dir = df.iloc[-1]['SUPERTd_10_3.0']
    current_trend = df.iloc[-1]['SUPERT_10_3.0']
    past_trend = df.iloc[-2]['SUPERT_10_3.0']
    rsi = df.iloc[-1]['RSI']

    # Calculate pivots
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name, 10, 10), axis=1)
    # Find last pivot high before current candle
    pivot_highs = df[df['pivot'] == 2]
    if not pivot_highs.empty:
        latest_pivot_high = pivot_highs.iloc[-1]['high']
    else:
        latest_pivot_high = None
    # Find last pivot low before current candle
    pivot_lows = df[df['pivot'] == 1]
    if not pivot_lows.empty:
        latest_pivot_low = pivot_lows.iloc[-1]['low']
    else:
        latest_pivot_low = None

    current_price = df.iloc[-1]['close']

    # Prevent trading in overbought/oversold markets
    if rsi > 70 or rsi < 30:
        global_value.logger(f"⏭️ Skipping trade due to RSI ({rsi:.2f}) being overbought/oversold.", "INFO")
        return None

    # Add trend check: skip if current trend != past trend
    if current_trend == past_trend:
        global_value.logger(f"⏭️ Skipping trade due to flat trend (current: {current_trend}, past: {past_trend})", "INFO")
        return None

    if call_conf > PROB_THRESHOLD:
        if latest_dir == 1 and latest_pivot_high is not None and current_price < latest_pivot_high:
            decision = "call"
            emoji = "🟢"
            confidence = call_conf
        else:
            global_value.logger(f"⏭️ Skipping CALL ({call_conf:.2%}) due to trend mismatch or price above pivot high.", "INFO")
            return None
    elif put_conf > PROB_THRESHOLD:
        if latest_dir == -1 and latest_pivot_low is not None and current_price > latest_pivot_low:
            decision = "put"
            emoji = "🔴"
            confidence = put_conf
        else:
            global_value.logger(f"⏭️ Skipping PUT ({put_conf:.2%}) due to trend mismatch or price below pivot low.", "INFO")
            return None
    else:
        if call_conf > put_conf:
            global_value.logger(f"⏭️ Skipping CALL due to low confidence ({call_conf:.2%})", "INFO")
        else:
            global_value.logger(f"⏭️ Skipping PUT due to low confidence ({put_conf:.2%})", "INFO")
        return None

    global_value.logger(f"{emoji} === PREDICTED: {decision.upper()} | CONFIDENCE: {confidence:.2%}", "INFO")
    return decision

def perform_trade(amount, pair, action, expiration):
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    trade_id = result[1]

    if result[0] is False or trade_id is None:
        global_value.logger("❗Trade failed to execute. Attempting reconnection...", "ERROR")
        api.disconnect()
        time.sleep(2)
        api.connect()
        return None

    time.sleep(expiration)
    return api.check_win(trade_id)

def martingale_strategy(pair, action):
    global current_profit

    amount = INITIAL_AMOUNT
    level = 1
    result = perform_trade(amount, pair, action, expiration)

    if result is None:
        return

    while result[1] == 'loose' and level < MARTINGALE_LEVEL:
        level += 1
        amount *= 2
        result = perform_trade(amount, pair, action, expiration)

        if result is None:
            return
        
    if result[1] != 'loose':
        global_value.logger("WIN - Resetting to base amount.", "INFO")
    else:
        global_value.logger("LOSS. Resetting.", "INFO")

def wait_until_next_candle(period_seconds=300, seconds_before=30):
    while True:
        now = datetime.now(timezone.utc)
        next_candle = ((now.timestamp() // period_seconds) + 1) * period_seconds
        if now.timestamp() >= next_candle - seconds_before:
            break
        time.sleep(0.2)

def wait_for_candle_start():
    while True:
        now = datetime.now(timezone.utc)
        if now.second == 0 and now.minute % (period // 60) == 0:
            break
        time.sleep(0.1)

def main_trading_loop():
    while True:
        global_value.logger("🔄 Starting new trading cycle...", "INFO")

        if not get_payout():
            global_value.logger("❗Failed to get payout data.", "ERROR")
            time.sleep(5)
            continue

        wait_until_next_candle(period_seconds=period, seconds_before=30)
        global_value.logger("🕒 30 seconds before candle. Preparing data and predictions...", "INFO")

        selected_pair = None
        selected_action = None

        for pair in list(global_value.pairs.keys()):
            oanda_pair = pair[:3] + "_" + pair[3:]
            df = get_oanda_candles(oanda_pair)

            if df is None:
                continue

            df = prepare_data(df)
            decision = train_and_predict(df)

            if decision:
                selected_pair = pair
                selected_action = decision
                global_value.logger(f"✅ Selected {pair} for {decision.upper()} trade.", "INFO")
                break  # Stop at first valid signal

        wait_for_candle_start()

        if selected_pair and selected_action:
            global_value.logger(f"🚀 Executing trade on {selected_pair} - {selected_action.upper()}", "INFO")
            martingale_strategy(selected_pair, selected_action)
        else:
            global_value.logger("⛔ No valid trading signal this cycle.", "INFO")

        # Optional: small pause before starting next cycle
        time.sleep(2)

if __name__ == "__main__":
    main_trading_loop()






