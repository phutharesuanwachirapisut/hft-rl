import polars as pl
from pathlib import Path

def generate_l2_microstructure_features(l2_parquet_path: str, levels: int = 5) -> pl.DataFrame:
    """
    Reads L2 Order Book Data lazily and computes OFI and OBI.
    Optimized for memory efficiency on Apple Silicon.
    """
    print(f"🧠 Calculating L2 Microstructure features from {l2_parquet_path}...")
    
    # 1. โหลดข้อมูลแบบ Lazy (ไม่กิน RAM)
    lf = pl.scan_parquet(l2_parquet_path)
    
    # 2. คำนวณ OBI (Order Book Imbalance) เชิงลึก n levels
    # สร้าง Expression เพื่อ Sum Volume ของฝั่ง Bid และ Ask ตั้งแต่ Level 1 ถึง n
    bid_vol_expr = pl.sum_horizontal([pl.col(f"b{i}_q") for i in range(1, levels + 1)])
    ask_vol_expr = pl.sum_horizontal([pl.col(f"a{i}_q") for i in range(1, levels + 1)])
    
    lf = lf.with_columns([
        ((bid_vol_expr - ask_vol_expr) / (bid_vol_expr + ask_vol_expr + 1e-8)).alias("obi")
    ])
    
    # 3. คำนวณ OFI (Order Flow Imbalance) ที่ Best Bid/Ask (Level 1)
    # เตรียมค่าจาก Step ก่อนหน้า (t-1)
    lf = lf.with_columns([
        pl.col("b1_p").shift(1).alias("prev_b1_p"),
        pl.col("b1_q").shift(1).alias("prev_b1_q"),
        pl.col("a1_p").shift(1).alias("prev_a1_p"),
        pl.col("a1_q").shift(1).alias("prev_a1_q")
    ])
    
    # คำนวณ e_t (Bid side update)
    e_t = (
        pl.when(pl.col("b1_p") > pl.col("prev_b1_p")).then(pl.col("b1_q"))
        .when(pl.col("b1_p") == pl.col("prev_b1_p")).then(pl.col("b1_q") - pl.col("prev_b1_q"))
        .otherwise(-pl.col("prev_b1_q"))
    )
    
    # คำนวณ f_t (Ask side update)
    f_t = (
        pl.when(pl.col("a1_p") < pl.col("prev_a1_p")).then(pl.col("a1_q"))
        .when(pl.col("a1_p") == pl.col("prev_a1_p")).then(pl.col("a1_q") - pl.col("prev_a1_q"))
        .otherwise(-pl.col("prev_a1_q"))
    )
    
    # ประกอบร่างเป็น OFI และทำ Smoothing (Rolling Sum) เพื่อลด Noise ในระดับ Tick
    lf = lf.with_columns([
        (e_t - f_t).alias("ofi_tick")
    ]).with_columns([
        pl.col("ofi_tick").rolling_sum(window_size=10).alias("ofi_rolling_10")
    ])
    
    # 4. เลือกคอลัมน์ที่จำเป็นและประมวลผล (Streaming Execution)
    # เราทิ้งคอลัมน์ L2 ดิบทั้งหมดเพื่อประหยัด RAM ตอนเข้า RL Environment
    print("⏳ Executing lazy graph for L2 Features...")
    df_features = (
        lf.select([
            "timestamp", 
            "b1_p", "a1_p", # เก็บ Best Bid/Ask ไว้คำนวณ Mid-price
            "obi", 
            "ofi_rolling_10"
        ])
        .drop_nulls()
        .collect(streaming=True)
    )
    
    print(f"✅ L2 Features generated! Shape: {df_features.shape}")
    return df_features

# ==========================================
# Example Integration
# ==========================================
if __name__ == "__main__":
    # สมมติว่ามีไฟล์ L2 Snapshot
    # L2_PATH = "../../data/processed/BTCUSDT_l2_depth.parquet"
    # df_l2_features = generate_l2_microstructure_features(L2_PATH, levels=5)
    pass