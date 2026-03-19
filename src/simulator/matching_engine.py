import numpy as np
from numba import njit

@njit
def run_fast_matching_engine(
    agent_bid_price: float,
    agent_ask_price: float,
    agent_order_size: float,
    market_trades: np.ndarray,
    current_inventory: float,
    max_inventory: float,
    maker_fee: float
):
    """
    🏎️ High-Speed Numba Matching Engine (Strict Risk Limits Edition)
    - market_trades shape: (N, 3) -> [price, volume, side(1=Buy, -1=Sell)]
    """
    new_inventory = current_inventory
    cash_flow = 0.0
    bid_filled = False
    ask_filled = False
    
    # 🛡️ Epsilon: ป้องกันปัญหา Floating Point Math (เช่น 0.0003+0.0001 > 0.0004)
    epsilon = 1e-8

    # วนลูปอ่าน Trade ของตลาดทีละ Tick (Time-priority simulation)
    for i in range(len(market_trades)):
        trade_price = market_trades[i, 0]
        trade_vol = market_trades[i, 1]
        trade_side = market_trades[i, 2]

        # ----------------------------------------------------
        # 🔵 1. ฝั่ง BID (เราตั้งรอซื้อ -> ตลาดต้องสาดขายใส่เรา)
        # ----------------------------------------------------
        # ถ้าตลาดยังไม่เคาะซื้อไม้เรา (not bid_filled) + ตลาดสาดขาย (side == -1) + ราคาทุบลงมาถึง Bid ของเรา
        if not bid_filled and trade_side == -1.0 and trade_price <= agent_bid_price:
            
            # 🚨 STRICT RISK CHECK: ซื้อแล้วของต้องไม่ล้นมือ (Long Limit)
            if (new_inventory + agent_order_size) <= (max_inventory + epsilon):
                new_inventory += agent_order_size
                # เงินสดไหลออก = (ขนาดไม้ * ราคาที่ได้) * (1 - ค่าธรรมเนียม)
                cash_flow -= (agent_order_size * agent_bid_price) * (1.0 - maker_fee)
                bid_filled = True

        # ----------------------------------------------------
        # 🔴 2. ฝั่ง ASK (เราตั้งรอขาย -> ตลาดต้องเคาะซื้อใส่เรา)
        # ----------------------------------------------------
        # ถ้าตลาดยังไม่กวาดไม้เรา (not ask_filled) + ตลาดเคาะซื้อ (side == 1) + ราคาลากขึ้นมาถึง Ask ของเรา
        if not ask_filled and trade_side == 1.0 and trade_price >= agent_ask_price:
            
            # 🚨 STRICT RISK CHECK: ขายแล้วต้องไม่เกินลิมิต Short (Short Limit)
            if (new_inventory - agent_order_size) >= -(max_inventory + epsilon):
                new_inventory -= agent_order_size
                # เงินสดไหลเข้า = (ขนาดไม้ * ราคาที่ขายได้) * (1 - ค่าธรรมเนียม)
                cash_flow += (agent_order_size * agent_ask_price) * (1.0 - maker_fee)
                ask_filled = True

        # Early Exit Optimization: ถ้า Match ครบทั้ง 2 ฝั่งแล้ว (ครบรอบทำกำไร Spread) 
        # ให้เบรก Loop ทันทีเพื่อประหยัด CPU สำหรับ High-Frequency Data
        if bid_filled and ask_filled:
            break

    return new_inventory, cash_flow, bid_filled, ask_filled