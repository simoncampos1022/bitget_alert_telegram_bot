import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import threading

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ===== Constants =====
BITGET_TICKERS_URL = "https://api.bitget.com/api/v2/mix/market/tickers"
BITGET_CANDLE_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
PRODUCT_TYPE = "USDT-FUTURES"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1H"


# ===== Helpers: Bitget API =====
def get_current_price(symbol: str) -> float:
    try:
        params = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
        }
        resp = requests.get(BITGET_TICKERS_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget tickers error: {data.get('msg')}")
        for t in data.get("data", []):
            if t.get("symbol") == symbol:
                return float(t["lastPr"])  # type: ignore[index]
        raise RuntimeError(f"Symbol {symbol} not found in tickers response")
    except Exception as e:
        print(f"[REST] Price fetch error for {symbol}: {e}")
        return float("nan")


def fetch_candles(symbol: str, interval: str, limit: int) -> List[Dict]:
    params = {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "granularity": interval,
        "limit": limit,
    }
    try:
        resp = requests.get(BITGET_CANDLE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget candles error: {data.get('msg')}")
        rows = data.get("data", [])
        candles: List[Dict] = []
        for r in rows:
            candles.append({
                "timestamp": datetime.fromtimestamp(int(r[0]) / 1000, timezone.utc),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "quote_volume": float(r[6]),
            })
        candles.sort(key=lambda x: x["timestamp"])  # ascending
        return candles
    except Exception as e:
        print(f"[DATA] Error fetching candles for {symbol}: {e}")
        return []


# ===== Indicators =====
def calculate_fs_tr(df: pd.DataFrame, fs_length: int = 10) -> pd.DataFrame:
    high = df["high"].values
    low = df["low"].values
    median = (high + low) / 2
    value = np.zeros(len(median))
    fs = np.zeros(len(median))
    tr = np.zeros(len(median))

    if len(median) == 0:
        df["fs"] = []
        df["tr"] = []
        return df

    value[0] = 0
    fs[0] = 0
    tr[0] = 0

    for i in range(1, len(median)):
        if i < fs_length:
            window = median[: i + 1]
            max_h = float(np.max(window))
            min_l = float(np.min(window))
        else:
            window = median[i - fs_length + 1 : i + 1]
            max_h = float(np.max(window))
            min_l = float(np.min(window))

        if max_h != min_l:
            val = 0.33 * 2 * ((median[i] - min_l) / (max_h - min_l) - 0.5) + 0.67 * value[i - 1]
            val = np.clip(val, -0.999, 0.999)
            value[i] = val
            fs[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fs[i - 1]
            tr[i] = fs[i - 1]
        else:
            value[i] = 0
            fs[i] = fs[i - 1]
            tr[i] = tr[i - 1]

    df["fs"] = fs
    df["tr"] = tr
    return df


# ===== Signal logic =====
def get_symbol_signal(df: pd.DataFrame) -> Tuple[str, float, float]:
    # Returns (signal_text, latest_close, prev_close)
    if df.empty or len(df) < 2:
        return ("N/A", float("nan"), float("nan"))
    fs_now = float(df["fs"].iloc[-1])
    tr_now = float(df["tr"].iloc[-1])
    fs_prev = float(df["fs"].iloc[-2])
    tr_prev = float(df["tr"].iloc[-2])

    cross_up = (fs_prev < tr_prev) and (fs_now > tr_now)
    cross_down = (fs_prev > tr_prev) and (fs_now < tr_now)

    if cross_up:
        sig = "Cross Up"
    elif cross_down:
        sig = "Cross Down"
    else:
        sig = "Increasing" if fs_now > tr_now else "Decreasing"

    return (sig, float(df["close"].iloc[-1]), float(df["close"].iloc[-2]))


# ===== Telegram =====
def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        if not resp.ok:
            print(f"[TG] Failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[TG] Error sending message: {e}")


def format_message(now: datetime, signals: Dict[str, Tuple[str, float, float]]) -> str:
    # Emojis for color cue
    green = "🟢"
    red = "🔴"

    lines_sig: List[str] = []
    lines_price: List[str] = []

    for sym, (sig, last_close, prev_close) in signals.items():
        # Signal color
        if sig == "Cross Up":
            sig_disp = f"{green} Cross Up"
        elif sig == "Cross Down":
            sig_disp = f"{red} Cross Down"
        elif sig == "Increasing":
            sig_disp = "Increasing"
        elif sig == "Decreasing":
            sig_disp = "Decreasing"
        else:
            sig_disp = sig

        lines_sig.append(f"{sym}: {sig_disp}")

        # Price section with color based on change vs 1h ago
        if not (np.isnan(last_close) or np.isnan(prev_close)):
            price_color = green if last_close > prev_close else red if last_close < prev_close else ""  # neutral
            price_text = f"{price_color} {last_close:,.2f}" if price_color else f"{last_close:,.2f}"
        else:
            price_text = "N/A"
        lines_price.append(f"{sym}: {price_text}")

    ts = now.strftime("%Y.%m.%d %H:%M")
    msg = (
        f"<b>Signal Section</b>\n"
        f"Current Time is {ts} UTC\n"
        + "\n".join(lines_sig)
        + "\n\n<b>Price Section</b>\n"
        + "\n".join(lines_price)
    )
    return msg


def run_once(token: str, chat_id: str) -> None:
    now = datetime.now(timezone.utc)
    signals: Dict[str, Tuple[str, float, float]] = {}
    for sym in SYMBOLS:
        candles = fetch_candles(sym, INTERVAL, 200)
        if not candles:
            signals[sym] = ("N/A", float("nan"), float("nan"))
            continue
        df = pd.DataFrame(candles)
        df = calculate_fs_tr(df, 10)
        sig = get_symbol_signal(df)
        signals[sym] = sig

    message = format_message(now, signals)
    send_telegram_message(token, chat_id, message)


def sleep_until_next_hour_plus_5s():
    now = datetime.now(timezone.utc)
    target = now.replace(minute=0, second=5, microsecond=0)
    if target <= now:
        target = target + timedelta(hours=1)
    sec = (target - now).total_seconds()
    print(f"Sleeping {sec:.2f}s until {target.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    time.sleep(sec)


def format_price_only(now: datetime) -> str:
    green = "🟢"
    red = "🔴"
    lines_price: List[str] = []
    for sym in SYMBOLS:
        current = get_current_price(sym)
        ref_close = float("nan")
        candles = fetch_candles(sym, INTERVAL, 2)
        if candles:
            # Use last closed 1H candle as reference
            ref_close = candles[-1]["close"]
        if not (np.isnan(current) or np.isnan(ref_close)):
            color = green if current > ref_close else red if current < ref_close else ""
            price_text = f"{color} {current:,.2f}" if color else f"{current:,.2f}"
        else:
            price_text = "N/A"
        lines_price.append(f"{sym}: {price_text}")

    ts = now.strftime("%Y.%m.%d %H:%M")
    return f"<b>Price Snapshot</b>\nCurrent Time is {ts} UTC\n" + "\n".join(lines_price)


def command_listener(token: str, allowed_chat_id: str) -> None:
    """Poll Telegram for commands and respond. Currently supports /price."""
    base_url = f"https://api.telegram.org/bot{token}"
    update_offset = None
    while True:
        try:
            params = {"timeout": 20}
            if update_offset is not None:
                params["offset"] = update_offset
            r = requests.get(f"{base_url}/getUpdates", params=params, timeout=25)
            data = r.json()
            if not data.get("ok", False):
                time.sleep(2)
                continue
            for upd in data.get("result", []):
                update_offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                chat_id = str(chat.get("id"))
                text = (msg.get("text") or "").strip()
                if chat_id != str(allowed_chat_id):
                    # Ignore messages from other chats
                    continue
                if text.startswith("/price"):
                    now = datetime.now(timezone.utc)
                    reply = format_price_only(now)
                    send_telegram_message(token, chat_id, reply)
                elif text.startswith("/start"):
                    send_telegram_message(token, chat_id, "Send /price to get current prices.")
        except Exception:
            time.sleep(2)


def main():
    # Load .env from project root if present
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        missing = []
        if not token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        raise SystemExit(
            "Missing environment variables: " + ", ".join(missing) +
            "\nCreate a .env file in the project folder with:\n"
            "TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN\nTELEGRAM_CHAT_ID=YOUR_CHAT_ID"
        )

    print("[ALERT BOT] Starting hourly alert loop...")
    # Start command listener thread
    threading.Thread(target=command_listener, args=(token, chat_id), daemon=True).start()
    while True:
        try:
            print(f"[ALERT BOT] Tick at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
            run_once(token, chat_id)
        except Exception as e:
            print(f"[ALERT BOT] Error: {e}")
        finally:
            sleep_until_next_hour_plus_5s()


if __name__ == "__main__":
    main()


