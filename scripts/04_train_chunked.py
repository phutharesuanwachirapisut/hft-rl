import os
import sys
import glob
import random
import polars as pl
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# เพิ่ม Path ให้ Python รู้จักโฟลเดอร์ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulator.market_env import BinanceMarketMakerEnv

def load_chunk_to_numpy(parquet_path: str) -> np.ndarray:
    """โหลดข้อมูล 1 Chunk เข้า RAM"""
    df = pl.read_parquet(parquet_path)
    # feature_cols = ["price", "tfi", "volatility_60s", "vpin"] 
    feature_cols = ["tfi", "volatility_60s", "vpin"] 
    return df.select(feature_cols).to_numpy().astype(np.float32)

def main():
    print("🚀 Starting Chunked Training Pipeline (Aggressive Mode)...")
    
    CHUNK_DIR = "/Users/zone/Documents/Project/RL/data/processed/chunks/"
    chunk_files = glob.glob(os.path.join(CHUNK_DIR, "*.parquet"))
    
    if not chunk_files:
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ Chunk ใน {CHUNK_DIR}")
        
    print(f"📁 Found {len(chunk_files)} chunks. Preparing curriculum...")

    # ==========================================
    # ⚙️ อัปเดต Hyperparameters (The Aggressive Tuning)
    # ==========================================
    config = {
        "initial_balance": 30.0,
        "max_inventory": 0.0004,
        "order_size": 0.0001,
        "eta": 10000.0,        # 📉 ลดลงมาจาก 200,000 ให้บอทกล้าถือของบ้าง
        "min_spread": 2.0,
        "max_spread": 60.0,    
        "vol_multiplier": 20.0,
        "max_skew_usd": 30.0   
    }
    
    os.makedirs("/Users/zone/Documents/Project/RL/models", exist_ok=True)
    MODEL_SAVE_PATH = "/Users/zone/Documents/Project/RL/models/ppo_hft_chunked"
    
    # 💥 เพิ่มเวลาเรียนรู้ให้ AI เพราะข้อสอบยากขึ้น
    TOTAL_EPOCHS = 10 
    
    model = None 

    for epoch in range(1, TOTAL_EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"🎓 เริ่มต้น EPOCH ที่ {epoch}/{TOTAL_EPOCHS}")
        print(f"{'='*40}")
        
        random.shuffle(chunk_files)
        
        for chunk_idx, chunk_path in enumerate(chunk_files):
            chunk_name = os.path.basename(chunk_path)
            print(f"\n📥 [Epoch {epoch}] Loading Chunk {chunk_idx+1}/{len(chunk_files)}: {chunk_name}")
            
            np_data = load_chunk_to_numpy(chunk_path)
            env = DummyVecEnv([lambda: BinanceMarketMakerEnv(data=np_data, config=config)])
            
            if model is None:
                print("🧠 กำลังสร้าง PPO Agent ตัวใหม่...")
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    learning_rate=0.00005, # 📉 เติม 0 ไปอีกตัว
                    n_steps=4096,
                    batch_size=512,
                    ent_coef=0.05,
                    device="cpu", 
                    verbose=1
                )
            else:
                print("🧠 อัปเดต Environment ใหม่ให้ Agent ตัวเดิม...")
                model.set_env(env)
            
            steps_in_chunk = len(np_data)
            print(f"🔥 Training on {steps_in_chunk} steps...")
            
            model.learn(total_timesteps=steps_in_chunk, reset_num_timesteps=False)
            model.save(f"{MODEL_SAVE_PATH}_latest")
            print(f"💾 Saved Checkpoint: {MODEL_SAVE_PATH}_latest")
            
            del np_data
            del env

    print("\n✅✅✅ Aggressive Chunked Training Complete! ✅✅✅")
    model.save(f"{MODEL_SAVE_PATH}_final")

if __name__ == "__main__":
    main()