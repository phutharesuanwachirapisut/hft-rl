import polars as pl
from pathlib import Path

def process_raw_trades_to_parquet(raw_csv_path: str, output_parquet_path: str) -> None:
    """
    Reads pre-processed Pandas CSV lazily, casts types to optimize memory, 
    and streams directly to a compressed Parquet file.
    """
    print(f"🔄 Scanning data from: {raw_csv_path}...")
    
    # 1. ปรับ Schema ให้ตรงกับไฟล์ CSV ที่ออกมาจาก Jupyter Notebook ของคุณ
    dtypes = {
        'aggregate_trade_id': pl.Int64,
        'price': pl.Float64,        
        'quantity': pl.Float64,
        'first_trade_id': pl.Int64,
        'last_trade_id': pl.Int64,
        'timestamp': pl.Int64,
        'is_buyer_maker': pl.Boolean,
        'is_best_match': pl.Boolean,
        'datetime': pl.String,  # Pandas เซฟเวลาเป็น String ลง CSV
        'side': pl.String       # คุณแมป 'BUY' / 'SELL' ไว้แล้วใน Notebook
    }

    # 2. อ่านไฟล์โดยเปิดโหมด has_header=True
    lf = pl.scan_csv(
        raw_csv_path,
        has_header=True, 
        dtypes=dtypes
    )

    # 3. Data Transformation (Lazy Execution)
    lf_transformed = (
        lf
        .with_columns([
            # แม้ใน CSV จะมีคอลัมน์ datetime (String) อยู่แล้ว 
            # แต่การใช้ timestamp มาแปลงใหม่เป็น Polars Datetime จะประมวลผลเร็วกว่าการ Parse String มาก
            pl.from_epoch(pl.col("timestamp"), time_unit="us").alias("datetime")
        ])
        # Sort data ตามเวลา เพื่อความแน่ใจในการทำ Time-series
        .sort("timestamp")
    )

    # 4. Stream to Parquet (The Magic for 16GB RAM constraints)
    print(f"💾 Streaming data to Parquet: {output_parquet_path}...")
    
    lf_transformed.sink_parquet(
        output_parquet_path,
        compression="zstd",
        row_group_size=100000 
    )
    
    print("✅ Parquet generation complete!")

if __name__ == "__main__":
    RAW_FILE = "../../data/raw/BTCUSDT_AggTrades.csv"
    PROCESSED_FILE = "../../data/processed/BTCUSDT_tick.parquet"
    # process_raw_trades_to_parquet(RAW_FILE, PROCESSED_FILE)