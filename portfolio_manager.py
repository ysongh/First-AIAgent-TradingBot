import os, json, time, math, requests, schedule, openai
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
from datetime import datetime
 
load_dotenv()                                     # read .env
 
# ------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------
RECALL_KEY  = os.getenv("RECALL_API_KEY")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")         # may be None
SANDBOX_API = "https://api.competitions.recall.network"
 
TOKEN_MAP = {                                     # main‑net addresses (sandbox forks main‑net)
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # 6 dec
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # 18 dec
}
 
DECIMALS = {"USDC": 6, "WETH": 18}
 
COINGECKO_IDS = {                                 # symbol → CG id
    "USDC": "usd-coin",
    "WETH": "weth",
}
 
DRIFT_THRESHOLD = 0.02    # rebalance if > 2 % off target
REB_TIME        = "09:00" # local server time

# New profit/loss thresholds
PROFIT_THRESHOLD = 10.0   # Take profits if gain > $10
LOSS_THRESHOLD = 50.0     # Stop loss if loss > $50
 
# ------------------------------------------------------------
#  Helper utilities
# ------------------------------------------------------------
def load_targets() -> dict[str, float]:
    with open("portfolio_config.json") as f:
        return json.load(f)

def load_position_history() -> dict:
    """Load historical position data for P&L tracking"""
    try:
        with open("position_history.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_position_history(history: dict):
    """Save position history to file"""
    with open("position_history.json", "w") as f:
        json.dump(history, f, indent=2)

def update_position_history(symbol: str, side: str, amount: float, price: float, timestamp: str):
    """Update position history with new trade"""
    history = load_position_history()
    
    if symbol not in history:
        history[symbol] = {
            "positions": [],
            "total_cost_basis": 0.0,
            "total_quantity": 0.0
        }
    
    position_data = history[symbol]
    
    if side == "buy":
        # Add to position
        cost = amount * price
        position_data["total_cost_basis"] += cost
        position_data["total_quantity"] += amount
        position_data["positions"].append({
            "type": "buy",
            "amount": amount,
            "price": price,
            "cost": cost,
            "timestamp": timestamp
        })
    elif side == "sell":
        # Remove from position (FIFO)
        if position_data["total_quantity"] > 0:
            avg_cost_basis = position_data["total_cost_basis"] / position_data["total_quantity"]
            cost_reduction = amount * avg_cost_basis
            position_data["total_cost_basis"] = max(0, position_data["total_cost_basis"] - cost_reduction)
            position_data["total_quantity"] = max(0, position_data["total_quantity"] - amount)
            
            position_data["positions"].append({
                "type": "sell",
                "amount": amount,
                "price": price,
                "revenue": amount * price,
                "timestamp": timestamp
            })
    
    save_position_history(history)

def calculate_pnl(symbol: str, current_price: float, current_holdings: float) -> dict:
    """Calculate unrealized P&L for a position"""
    history = load_position_history()
    
    if symbol not in history or current_holdings <= 0:
        return {"unrealized_pnl": 0.0, "avg_cost_basis": 0.0}
    
    position_data = history[symbol]
    
    if position_data["total_quantity"] <= 0:
        return {"unrealized_pnl": 0.0, "avg_cost_basis": 0.0}
    
    avg_cost_basis = position_data["total_cost_basis"] / position_data["total_quantity"]
    current_value = current_holdings * current_price
    cost_value = current_holdings * avg_cost_basis
    unrealized_pnl = current_value - cost_value
    
    return {
        "unrealized_pnl": unrealized_pnl,
        "avg_cost_basis": avg_cost_basis,
        "current_value": current_value,
        "cost_value": cost_value
    }
 
def to_base_units(amount_float: float, decimals: int) -> str:
    """Convert human units → integer string that Recall expects."""
    scaled = Decimal(str(amount_float)) * (10 ** decimals)
    return str(int(scaled.quantize(Decimal("1"), rounding=ROUND_DOWN)))
 
# ------------------------------------------------------------
#  Market data
# ------------------------------------------------------------
def fetch_prices(symbols: list[str]) -> dict[str, float]:
    ids = ",".join(COINGECKO_IDS[sym] for sym in symbols)
    r = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": ids, "vs_currencies": "usd"},
        timeout=10,
    )
    data = r.json()
    return {sym: data[COINGECKO_IDS[sym]]["usd"] for sym in symbols}
 
def fetch_holdings() -> dict[str, float]:
    """Return whole‑token balances from Recall's sandbox."""
    r = requests.get(
        f"{SANDBOX_API}/api/agent/balances",
        headers={"Authorization": f"Bearer {RECALL_KEY}"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()

    print("✅  Balances:", data)

    # Parse the nested response structure
    holdings = {}
    for balance in data.get('balances', []):
        symbol = balance['symbol']
        amount = balance['amount']
        
        # Aggregate balances by symbol (in case there are multiple chains)
        if symbol in holdings:
            holdings[symbol] += amount
        else:
            holdings[symbol] = amount
    
    return holdings        # → {"USDC": 123.45, ...}

# ------------------------------------------------------------
#  Profit/Loss Management
# ------------------------------------------------------------
def check_pnl_triggers(prices: dict, holdings: dict) -> list:
    """Check if any positions trigger profit-taking or stop-loss"""
    pnl_orders = []
    timestamp = datetime.now().isoformat()
    
    for symbol in holdings:
        if symbol == "USDC" or holdings[symbol] <= 0:
            continue
            
        current_price = prices.get(symbol, 0)
        if current_price <= 0:
            continue
            
        pnl_data = calculate_pnl(symbol, current_price, holdings[symbol])
        unrealized_pnl = pnl_data["unrealized_pnl"]
        
        print(f"📊 {symbol} P&L: ${unrealized_pnl:.2f} (Avg cost: ${pnl_data.get('avg_cost_basis', 0):.2f}, Current: ${current_price:.2f})")
        
        # Check for profit-taking trigger
        if unrealized_pnl >= PROFIT_THRESHOLD:
            print(f"💰 Profit trigger hit for {symbol}: +${unrealized_pnl:.2f}")
            pnl_orders.append({
                "symbol": symbol,
                "side": "sell",
                "amount_float": holdings[symbol],  # Sell entire position
                "reason": f"profit_taking",
                "pnl": unrealized_pnl,
                "trigger_type": "profit"
            })
            
        # Check for stop-loss trigger  
        elif unrealized_pnl <= -LOSS_THRESHOLD:
            print(f"🛑 Stop-loss trigger hit for {symbol}: -${abs(unrealized_pnl):.2f}")
            pnl_orders.append({
                "symbol": symbol,
                "side": "sell", 
                "amount_float": holdings[symbol],  # Sell entire position
                "reason": f"stop_loss",
                "pnl": unrealized_pnl,
                "trigger_type": "stop_loss"
            })
    
    return pnl_orders
 
# ------------------------------------------------------------
#  Trading logic
# ------------------------------------------------------------
def compute_orders(targets, prices, holdings):
    """Return a list of {'symbol','side','amount_float','usd_value'} trades."""
    total_value = sum(holdings.get(s, 0) * prices[s] for s in targets)
    if total_value == 0:
        raise ValueError("No balances found; fund your sandbox wallet first.")
 
    overweight, underweight = [], []
    for sym, weight in targets.items():
        current_val = holdings.get(sym, 0) * prices[sym]
        target_val  = total_value * weight
        drift_pct   = (current_val - target_val) / total_value
        if abs(drift_pct) >= DRIFT_THRESHOLD:
            delta_val = abs(target_val - current_val)
            token_amt = delta_val / prices[sym]
            side      = "sell" if drift_pct > 0 else "buy"
            
            order = {
                "symbol": sym, 
                "side": side, 
                "amount_float": token_amt,
                "usd_value": delta_val,
                "reason": "rebalance",
                "trigger_type": "rebalance"
            }
            
            (overweight if side == "sell" else underweight).append(order)
 
    # Execute sells first so we have USDC to fund buys
    return overweight + underweight
 
def execute_trade(symbol, side, amount_float, usd_value=None, reason="trade", trigger_type="manual"):
    """Execute a trade and update position history"""
    # Skip USDC trades since we can't trade USDC for USDC
    if symbol == "USDC":
        print(f"⚠️ Skipping {side} {amount_float} USDC (cannot trade USDC for USDC)")
        return {"status": "skipped", "reason": "Cannot trade USDC for USDC"}
    
    from_token, to_token = (
        (TOKEN_MAP[symbol], TOKEN_MAP["USDC"]) if side == "sell"
        else (TOKEN_MAP["USDC"], TOKEN_MAP[symbol])
    )
 
    # For buy orders, amount should be in USDC (what we're spending)
    # For sell orders, amount should be in the token we're selling
    if side == "buy":
        # Use the USD value to determine how much USDC to spend
        amount_to_use = usd_value if usd_value else amount_float * 3500  # rough WETH price fallback
        decimals_to_use = DECIMALS["USDC"]
    else:
        amount_to_use = amount_float
        decimals_to_use = DECIMALS[symbol]
    
    payload = {
        "fromToken": from_token,
        "toToken":   to_token,
        "amount":    to_base_units(amount_to_use, decimals_to_use),
        "reason":    f"{reason} - {trigger_type}",
    }
    
    # Debug logging
    print(f"🔍 Trading {side} {amount_float} {symbol} ({reason})")
    print(f"🔍 Payload: {json.dumps(payload, indent=2)}")
    
    r = requests.post(
        f"{SANDBOX_API}/api/trade/execute",
        json=payload,
        headers={
            "Authorization": f"Bearer {RECALL_KEY}",
            "Content-Type":  "application/json",
        },
        timeout=20,
    )
    
    # Debug the response before raising for status
    print(f"🔍 Response Status: {r.status_code}")
    print(f"🔍 Response Body: {r.text}")
    
    if not r.ok:
        print(f"❌ Trade failed: {r.status_code} - {r.text}")
        return {"status": "failed", "error": r.text}
    
    # If trade was successful, update position history
    try:
        # Get current price for the symbol (rough estimate for history tracking)
        current_prices = fetch_prices([symbol])
        current_price = current_prices.get(symbol, 0)
        
        if current_price > 0:
            timestamp = datetime.now().isoformat()
            update_position_history(symbol, side, amount_float, current_price, timestamp)
            print(f"📝 Updated position history for {symbol}")
    except Exception as e:
        print(f"⚠️ Failed to update position history: {e}")
    
    return r.json()
 
# ------------------------------------------------------------
#  Optional: GPT‑4o target adjustments
# ------------------------------------------------------------
def ai_adjust_targets(targets: dict[str, float]) -> dict[str, float]:
    if not OPENAI_KEY:
        return targets                           # AI disabled
    
    client = openai.OpenAI(api_key=OPENAI_KEY)
 
    prompt = (
        "Here is my current target allocation (weights sum to 1):\n"
        f"{json.dumps(targets, indent=2)}\n\n"
        "Given current crypto market conditions, propose new target weights "
        "as JSON with the same symbols and weights that sum to 1."
    )
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = chat.choices[0].message.content
    try:
        # Remove triple‑backtick blocks if model returns Markdown
        clean = raw.strip("` \n")
        return json.loads(clean)
    except json.JSONDecodeError:
        print("⚠️  GPT response was not valid JSON, keeping existing targets")
        return targets
 
# ------------------------------------------------------------
#  Enhanced Daily job with P&L monitoring
# ------------------------------------------------------------
def rebalance():
    targets   = load_targets()
    targets   = ai_adjust_targets(targets)
    prices    = fetch_prices(list(targets.keys()))
    holdings  = fetch_holdings()
    
    # Debug logging
    print(f"🔍 Targets: {targets}")
    print(f"🔍 Prices: {prices}")
    print(f"🔍 Holdings: {holdings}")
    
    # First, check for P&L triggers (profit-taking and stop-loss)
    pnl_orders = check_pnl_triggers(prices, holdings)
    
    if pnl_orders:
        print(f"🎯 Executing {len(pnl_orders)} P&L triggered orders first...")
        for order in pnl_orders:
            print(f"🔄 Executing {order['trigger_type']} order: {order}")
            res = execute_trade(
                order["symbol"], 
                order["side"], 
                order["amount_float"], 
                reason=order["reason"],
                trigger_type=order["trigger_type"]
            )
            print(f"✅ {order['trigger_type'].title()} executed for {order['symbol']}: P&L ${order['pnl']:.2f} → {res.get('status', 'unknown')}")
        
        # Refresh holdings after P&L trades
        holdings = fetch_holdings()
        print("🔄 Refreshed holdings after P&L trades")
    
    # Then proceed with normal rebalancing
    orders = compute_orders(targets, prices, holdings)
 
    if not orders:
        print("✅ Portfolio already within ±2% of target.")
    else:
        print(f"🔍 Generated {len(orders)} rebalancing orders")
        for order in orders:
            print(f"🔄 Executing rebalance order: {order}")
            res = execute_trade(
                order["symbol"], 
                order["side"], 
                order["amount_float"], 
                order.get("usd_value"),
                reason=order["reason"],
                trigger_type=order["trigger_type"]
            )
            print(f"✅ Rebalance executed for {order['symbol']} → {res.get('status', 'unknown')}")
 
    print("🎯 Rebalance complete.")

def run_pnl_check():
    """Separate function to run P&L checks more frequently"""
    try:
        targets = load_targets()
        prices = fetch_prices(list(targets.keys()))
        holdings = fetch_holdings()
        
        pnl_orders = check_pnl_triggers(prices, holdings)
        
        if pnl_orders:
            print(f"⚡ P&L Check: Executing {len(pnl_orders)} triggered orders...")
            for order in pnl_orders:
                res = execute_trade(
                    order["symbol"], 
                    order["side"], 
                    order["amount_float"], 
                    reason=order["reason"],
                    trigger_type=order["trigger_type"]
                )
                print(f"✅ {order['trigger_type'].title()} executed: {order['symbol']} P&L ${order['pnl']:.2f}")
        else:
            print("✅ P&L Check: No triggers hit")
            
    except Exception as e:
        print(f"❌ P&L check failed: {e}")
 
# ------------------------------------------------------------
#  Enhanced Scheduler
# ------------------------------------------------------------
# Schedule rebalancing once daily
schedule.every().day.at(REB_TIME).do(rebalance)

# Schedule P&L checks every 30 minutes during market hours (more frequent monitoring)
schedule.every(30).minutes.do(run_pnl_check)
 
if __name__ == "__main__":
    print("🚀 Starting enhanced portfolio manager with P&L monitoring… (Ctrl‑C to quit)")
    print(f"💰 Profit threshold: +${PROFIT_THRESHOLD}")
    print(f"🛑 Stop-loss threshold: -${LOSS_THRESHOLD}")
    print(f"🔄 Rebalance time: {REB_TIME}")
    print(f"📊 P&L checks: Every 30 minutes")
    
    rebalance()                 # run once at launch
    while True:
        schedule.run_pending()
        time.sleep(60)