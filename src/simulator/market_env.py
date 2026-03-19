import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

# สมมติว่าคุณมี Numba Matching Engine อยู่ที่เดิม
from .matching_engine import run_fast_matching_engine

class BinanceMarketMakerEnv(gym.Env):
    """
    HFT Market Making Environment - Armored Edition 🛡️
    - Dynamic Skew based on Inventory
    - Volatility-adjusted Spread
    - Hard Risk Limits (Circuit Breakers)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data: np.ndarray, config: Dict[str, Any]):
        super().__init__()
        
        # 1. Load Data (Expected columns: mid_price, volume, volatility_60s, vpin, obi, ofi_rolling_10)
        self.data = data
        self.n_steps = len(self.data)
        
        # 2. Market Config
        self.max_inventory = config.get("max_inventory", 0.0004)
        self.order_size = config.get("order_size", 0.0001)
        self.maker_fee = config.get("maker_fee", 0.0000)
        self.inventory_penalty = config.get("eta", 50000.0)
        
        # 🛡️ New Armor Configs
        self.min_spread = config.get("min_spread", 2.0)
        self.max_spread = config.get("max_spread", 40.0)
        self.vol_multiplier = config.get("vol_multiplier", 10.0) # ตัวคูณความผันผวน
        self.max_skew_usd = config.get("max_skew_usd", 20.0) # เพดาน Skew สูงสุด
        
        # 3. RL Spaces
        # Action: [Spread_Control, Skew_Aggressiveness (k)] 
        # ค่า Action จะอยู่ระหว่าง -1.0 ถึง 1.0 เสมอ
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Obs: [Market_Features(6) + Private_State(2)] = 8 Features
        n_features = self.data.shape[1] + 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        
        # 4. State Variables
        self.initial_balance = config.get("initial_balance", 30.0)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = self.initial_balance

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = self.initial_balance
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        market_obs = self.data[self.current_step]
        inv_ratio = self.inventory / self.max_inventory
        time_ratio = self.current_step / self.n_steps
        obs = np.append(market_obs, [inv_ratio, time_ratio])
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        mid_price = self.data[self.current_step, 0]
        mtm_balance = self.cash + (self.inventory * mid_price)
        return {
            "inventory": self.inventory,
            "cash": self.cash,
            "portfolio_value": mtm_balance,
            "pnl": mtm_balance - self.initial_balance
        }

    def _simulate_micro_trades(self, current_mid: float, next_mid: float, volume: float) -> np.ndarray:
        if next_mid > current_mid:
            side = 1.0 
        elif next_mid < current_mid:
            side = -1.0
        else:
            side = 1.0 if np.random.rand() > 0.5 else -1.0
            
        trades = np.array([
            [current_mid, volume * 0.3, side],
            [next_mid, volume * 0.7, side]
        ], dtype=np.float64)
        return trades

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= self.n_steps - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Extract Market Data
        mid_price = self.data[self.current_step, 0] 
        volume = self.data[self.current_step, 1]
        volatility = self.data[self.current_step, 2] # ดึง Volatility จาก State
        next_mid_price = self.data[self.current_step + 1, 0]

        # ==========================================
        # 🛡️ ARMOR 1 & 2: Dynamic Spread & Skew
        # ==========================================
        spread_action = action[0] # [-1, 1] บอทยังได้สิทธิ์คุม Spread ตามปกติ
        # k_action = action[1]    # ❌ ไม่สนใจค่า Skew จาก AI อีกต่อไป (ยึดอำนาจ)
        
        # 1. Spread = Base Spread (จาก AI) + Volatility Premium (กางออโต้ตามความผันผวน)
        base_half_spread = self.min_spread + ((spread_action + 1.0) / 2.0) * (self.max_spread - self.min_spread)
        vol_premium = volatility * self.vol_multiplier
        final_half_spread = base_half_spread + vol_premium
        
        # 2. Hardcoded Dynamic Skew (ยึดอำนาจมาใช้สมการคณิตศาสตร์ 100%)
        # ถ้าถือ Long (+0.0004) -> inv_ratio = 1.0 -> skew = +50 USD (กดราคา Bid/Ask ลงเพื่อเทขาย)
        # ถ้าถือ Short (-0.0004) -> inv_ratio = -1.0 -> skew = -50 USD (ดันราคา Bid/Ask ขึ้นเพื่อไล่ซื้อคืน)
        inv_ratio = self.inventory / self.max_inventory
        skew = inv_ratio * self.max_skew_usd
        
        my_bid = mid_price - final_half_spread - skew
        my_ask = mid_price + final_half_spread - skew
        # ==========================================
        # 🛡️ ARMOR 3: Hard Risk Limits (Circuit Breakers)
        # ==========================================
        # ชักปลั๊ก! ห้ามตั้ง Bid ถือของเต็ม Max แล้ว (ดักทาง Bug ใน Numba)
        if self.inventory >= self.max_inventory:
            my_bid = 0.0  # ตั้ง Bid เป็น 0 เพื่อไม่ให้เกิดการ Match เด็ดขาด
            
        # ห้ามตั้ง Ask ถ้าเปิด Short จนเต็มลิมิต (ถ้าบอทคุณเล่นฝั่ง Short ด้วย)
        if self.inventory <= -self.max_inventory:
            my_ask = float('inf') # ตั้ง Ask สูงเสียดฟ้า ไม่มีใครซื้อแน่นอน

        # ==========================================
        # Execute Matching
        # ==========================================
        simulated_trades = self._simulate_micro_trades(mid_price, next_mid_price, volume)
        
        new_inventory, cash_flow, bid_filled, ask_filled = run_fast_matching_engine(
            agent_bid_price=my_bid,
            agent_ask_price=my_ask,
            agent_order_size=self.order_size,
            market_trades=simulated_trades,
            current_inventory=self.inventory,
            max_inventory=self.max_inventory, # ย้ำ Max Inventory เข้าไปอีกรอบ
            maker_fee=self.maker_fee
        )

        # Mark-to-Market PnL
        old_portfolio_value = self.cash + (self.inventory * mid_price)
        
        self.inventory = new_inventory
        self.cash += cash_flow
        
        new_portfolio_value = self.cash + (self.inventory * next_mid_price)
        delta_pnl = new_portfolio_value - old_portfolio_value

        # Reward Calculation with heavy Penalty
        penalty = self.inventory_penalty * (self.inventory ** 2)
        reward = delta_pnl - penalty

        # Step Update & Stop-loss
        self.current_step += 1
        terminated = False
        
        # ถ้าพอร์ตหายไป 20% สั่งตัดจบ Episode (ตาย) 
        if new_portfolio_value < self.initial_balance * 0.8:
            terminated = True
            reward -= 1000.0 # โดนตีหัวแตกตอนจบ

        return self._get_observation(), float(reward), terminated, False, self._get_info()