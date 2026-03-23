import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple
import collections
from .matching_engine import run_fast_matching_engine

class BinanceMarketMakerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data: np.ndarray, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data = data
        self.n_steps = len(self.data)

        # 1. Config พื้นฐาน
        self.max_inventory = config.get("max_inventory", 0.0004)
        self.order_size = config.get("order_size", 0.0001)
        self.maker_fee = config.get("maker_fee", -0.0005) # ⭐️ ได้เงินทอน 5 bps
        self.eta = config.get("eta", 100.0)

        self.min_spread = config.get("min_spread", 2.0)
        self.max_spread = config.get("max_spread", 25.0)
        self.vol_multiplier = config.get("vol_multiplier", 10.0)
        self.max_skew_usd = config.get("max_skew_usd", 30.0)

        # 2. Action: [0] = Spread, [1] = Skew (ตรงกับสคริปต์ 03 เป๊ะๆ)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 3. Frame Stacking
        self.stack_size = config.get("frame_stack", 5)
        self.frames = collections.deque(maxlen=self.stack_size)

        # ⭐️ Features: vol, vola, tfi, vpin + inv, time = 6 ตัว
        self.n_features_per_frame = 6
        total_features = self.stack_size * self.n_features_per_frame
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)

        self.initial_balance = config.get("initial_balance", 30.0)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = self.initial_balance

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = self.initial_balance

        self.frames.clear()
        first_frame = self._get_current_frame()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)

        return self._get_stacked_observation(), self._get_info()

    def _get_current_frame(self) -> np.ndarray:
        # ดึง index 1 ถึง 4: volume, vola, tfi, vpin (ข้าม price ที่เป็น index 0)
        market_obs = self.data[self.current_step, 1:5]
        inv_ratio = self.inventory / self.max_inventory
        time_ratio = self.current_step / self.n_steps
        return np.append(market_obs, [inv_ratio, time_ratio]).astype(np.float32)

    def _get_stacked_observation(self) -> np.ndarray:
        return np.concatenate(list(self.frames))

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
        side = 1.0 if next_mid > current_mid else (-1.0 if next_mid < current_mid else (1.0 if np.random.rand() > 0.5 else -1.0))
        return np.array([
            [current_mid, volume * 0.3, side],
            [next_mid, volume * 0.7, side]
        ], dtype=np.float64)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= self.n_steps - 1:
            return self._get_stacked_observation(), 0.0, True, False, self._get_info()

        mid_price = self.data[self.current_step, 0]
        volume = self.data[self.current_step, 1]
        volatility = self.data[self.current_step, 2]
        tfi = self.data[self.current_step, 3]
        vpin = self.data[self.current_step, 4]
        next_mid_price = self.data[self.current_step + 1, 0]

        # ==========================================
        # 1. Action Decoding & Armor
        # ==========================================
        spread_action = action[0]
        skew_action = action[1]

        base_half_spread = self.min_spread + ((spread_action + 1.0) / 2.0) * (self.max_spread - self.min_spread)
        vol_premium = volatility * self.vol_multiplier
        final_half_spread = base_half_spread + vol_premium

        # ให้ AI คุม Skew ได้โดยตรง
        ai_skew = skew_action * self.max_skew_usd

        # ระบบนิรภัย: ช่วยดัมพ์ราคาเวลาของล้นมือ
        inv_ratio = self.inventory / self.max_inventory
        risk_skew = inv_ratio * self.max_skew_usd 
        final_skew = ai_skew + risk_skew # เอาความฉลาด AI ผสมกับระบบนิรภัย

        # ล็อคไม่ให้ Skew กว้างกว่า Spread
        max_safe_skew = final_half_spread - 0.5
        final_skew = np.clip(final_skew, -max_safe_skew, max_safe_skew)

        # VPIN Armor: ถ้ารายใหญ่เข้า (เกิน 80%) ถ่าง Spread หนี 10 USD
        if vpin > 0.8:
            final_half_spread += 10.0

        my_bid = mid_price - final_half_spread - final_skew
        my_ask = mid_price + final_half_spread - final_skew

        # Circuit Breakers
        if self.inventory >= self.max_inventory:
            my_bid = 0.0
        if self.inventory <= -self.max_inventory:
            my_ask = float('inf')

        # ==========================================
        # 2. Execute Matching
        # ==========================================
        simulated_trades = self._simulate_micro_trades(mid_price, next_mid_price, volume)

        new_inventory, cash_flow, bid_filled, ask_filled = run_fast_matching_engine(
            agent_bid_price=my_bid,
            agent_ask_price=my_ask,
            agent_order_size=self.order_size,
            market_trades=simulated_trades,
            current_inventory=self.inventory,
            max_inventory=self.max_inventory,
            maker_fee=self.maker_fee
        )

        old_portfolio_value = self.cash + (self.inventory * mid_price)
        self.inventory = new_inventory
        self.cash += cash_flow
        new_portfolio_value = self.cash + (self.inventory * next_mid_price)

        # ==========================================
        # 📈 THE PURE PNL REWARD (Asymmetric Risk Aversion)
        # ==========================================
        step_pnl = new_portfolio_value - old_portfolio_value
        inventory_penalty = self.eta * (self.inventory ** 2)
        
        # ⭐️ ยกเลิกโบนัสหลอกเด็ก (trade_bonus)
        # ปล่อยให้ Maker Rebate (-0.0005) ที่เราตั้งไว้ใน config ทำงานร่วมกับ PnL ไปตามธรรมชาติ

        # ⭐️ สมการใหม่: เกลียดการขาดทุนให้มากกว่าตอนได้กำไร (Risk Averse)
        if step_pnl < 0:
            # ถ้าก้าวนี้ขาดทุน ลงโทษหนักคูณ 2 (ให้มันกลัวการโดนลาก)
            reward = (step_pnl * 200.0) - inventory_penalty
        else:
            # ถ้าก้าวนี้กำไร (หรือเสมอตัว) ให้รางวัลปกติ
            reward = (step_pnl * 100.0) - inventory_penalty

        self.current_step += 1
        terminated = False

        # Stop Loss ที่ 50% ของพอร์ต
        if new_portfolio_value < self.initial_balance * 0.5:
            terminated = True
            reward -= 500.0 # ตีหัวแตกตอนพอร์ตละลาย

        if not terminated and self.current_step < self.n_steps:
            self.frames.append(self._get_current_frame())

        return self._get_stacked_observation(), float(reward), terminated, False, self._get_info()