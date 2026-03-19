import polars as pl
import numpy as np

def generate_rl_state_features(parquet_path: str, window_size: str = "1s") -> pl.DataFrame:
    """
    Reads Parquet lazily, resamples tick data into time buckets, 
    and calculates Microstructure features.
    """
    print(f"🧠 Calculating features from {parquet_path} with window {window_size}...")
    
    # อ่านไฟล์ Parquet แบบ Lazy
    lf = pl.scan_parquet(parquet_path)
    
    # 1. Resampling data using group_by_dynamic
    # ยุบ Tick data ให้กลายเป็น OHLCV และคำนวณ Imbalance
    lf_resampled = (
        lf
        .group_by_dynamic("datetime", every=window_size)
        .agg([
            # Price Features (ใช้ออเดอร์สุดท้ายของวิเป็นราคา Close)
            pl.col("price").last().alias("close_price"),
            
            # Volume Features
            pl.col("quantity").sum().alias("total_volume"),
            
            # Trade Flow Imbalance (TFI) 
            # เอา Volume ฝั่ง BUY ลบด้วย Volume ฝั่ง SELL
            (
                pl.when(pl.col("side") == "BUY").then(pl.col("quantity")).otherwise(0).sum() -
                pl.when(pl.col("side") == "SELL").then(pl.col("quantity")).otherwise(0).sum()
            ).alias("trade_flow_imbalance"),
            
            # นับจำนวนออเดอร์ (Trade Count)
            pl.count().alias("trade_count")
        ])
    )
    
    # 2. คำนวณ Rolling Volatility และ Features เชิงลึก
    lf_features = (
        lf_resampled
        # คำนวณ Log Return ของราคา Close
        .with_columns([
            (pl.col("close_price") / pl.col("close_price").shift(1))
            .log()
            .alias("log_return")
        ])
        # คำนวณ Rolling Volatility (เช่น 60 วินาที)
        .rolling(
            index_column="datetime",
            period="60s"
        )
        .agg([
            pl.col("close_price").last().alias("price"),
            pl.col("trade_flow_imbalance").last().alias("tfi"),
            pl.col("total_volume").last().alias("volume"),
            # Standard Deviation ของ Log Return ตลอด 60 วิที่ผ่านมา (Volatility)
            pl.col("log_return").std().alias("volatility_60s")
        ])
        # Drop แถวที่มี Null จากผลของ Rolling window ตอนเริ่มต้น
        .drop_nulls()
    )

    # 3. Execution!
    # ใช้ streaming=True เพื่อบังคับให้ Polars จัดการ Memory แบบ Out-of-core
    print("⏳ Executing lazy graph with streaming engine...")
    df_final = lf_features.collect(streaming=True)
    
    print(f"✅ Features generated! Shape: {df_final.shape}")
    return df_final

def calculate_vpin_and_merge(
    tick_parquet_path: str, 
    df_time_features: pl.DataFrame, 
    volume_bucket_size: float = 10.0, 
    window_size: int = 50
) -> pl.DataFrame:
    """
    คำนวณ VPIN ผ่าน Volume Buckets แล้ว Merge กลับเข้า Time-based Features
    """
    print(f"📊 Calculating VPIN (Bucket: {volume_bucket_size} BTC, Window: {window_size})...")
    
    # 1. โหลดข้อมูล Tick Data
    lf_ticks = pl.scan_parquet(tick_parquet_path)
    
    # 2. สร้าง Volume Clock (หั่นเป็นถังละ 10 BTC)
    # ใช้ cum_sum() เพื่อหา Volume สะสม แล้วหารด้วยขนาดถังเพื่อสร้าง ID ให้แต่ละถัง
    lf_buckets = (
        lf_ticks
        .with_columns([
            pl.col("quantity").cum_sum().alias("cum_volume")
        ])
        .with_columns([
            (pl.col("cum_volume") / volume_bucket_size).cast(pl.Int64).alias("bucket_id")
        ])
    )
    
    # 3. ยุบรวมข้อมูลในแต่ละถัง (Aggregating Volume Buckets)
    lf_vpin = (
        lf_buckets
        .group_by("bucket_id")
        .agg([
            pl.col("datetime").last().alias("datetime"), # ใช้เวลาที่ถังนี้ถูกเติมเต็มเป็นตัวแทน
            pl.when(pl.col("side") == "BUY").then(pl.col("quantity")).otherwise(0).sum().alias("buy_vol"),
            pl.when(pl.col("side") == "SELL").then(pl.col("quantity")).otherwise(0).sum().alias("sell_vol")
        ])
        .sort("bucket_id")
    )
    
    # 4. คำนวณสมการ VPIN
    df_vpin_signal = (
        lf_vpin
        .with_columns([
            (pl.col("buy_vol") - pl.col("sell_vol")).abs().alias("imbalance")
        ])
        .with_columns([
            pl.col("imbalance").rolling_sum(window_size=window_size).alias("rolling_imbalance")
        ])
        .with_columns([
            # หารด้วย Volume รวมของหน้าต่าง (n * V) เพื่อให้ค่า VPIN อยู่ระหว่าง 0 ถึง 1
            (pl.col("rolling_imbalance") / (window_size * volume_bucket_size)).alias("vpin")
        ])
        .select(["datetime", "vpin"])
        .drop_nulls()
        .collect() # โหลด VPIN ลง RAM เพราะข้อมูลถูกบีบอัดจนเล็กมากแล้ว
    )

    print(f"🔗 Merging VPIN to 1-second Time Features using As-Of Join...")
    
    # 5. As-of Join: แปะค่า VPIN เข้ากับแท่ง 1-second 
    # (ถ้าแท่งไหนไม่มี VPIN เกิดขึ้น ให้ดึงค่า VPIN ล่าสุดในอดีตมาใช้)
    df_final = df_time_features.join_asof(
        df_vpin_signal,
        on="datetime",
        strategy="backward"
    ).drop_nulls() # Drop แท่งแรกๆ ที่ VPIN ยังคำนวณไม่เสร็จ
    
    return df_final
    
# Example execution
if __name__ == "__main__":
    PROCESSED_FILE = "/Users/zone/Documents/Project/RL/data/processed/BTCUSDT_20260313_trades.parquet"
    df_state = generate_rl_state_features(PROCESSED_FILE)
    print(df_state.head())