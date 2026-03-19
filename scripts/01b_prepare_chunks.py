import os
import polars as pl

def main():
    print("🪓 Starting Domain Randomization Chunking...")
    OUTPUT_DIR = "/Users/zone/Documents/Project/RL/data/processed/chunks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    regimes = ["sideway", "trend", "toxic"]
    
    for regime in regimes:
        INPUT_FILE = f"/Users/zone/Documents/Project/RL/data/processed/BTCUSDT_features_{regime}.parquet"
        
        if not os.path.exists(INPUT_FILE):
            print(f"⚠️ ข้าม {regime} - ไม่พบไฟล์ {INPUT_FILE}")
            continue

        print(f"\n📥 Loading {regime} dataset...")
        df = pl.read_parquet(INPUT_FILE)
        
        df = df.with_columns([
            pl.col("datetime").dt.year().alias("year"),
            pl.col("datetime").dt.week().alias("week")
        ])
        
        unique_weeks = df.select(["year", "week"]).unique().sort(["year", "week"])
        print(f"🔪 Found {len(unique_weeks)} weeks in {regime}. Splitting...")
        
        for row in unique_weeks.iter_rows(named=True):
            y = row["year"]
            w = row["week"]
            
            df_chunk = df.filter((pl.col("year") == y) & (pl.col("week") == w))
            df_chunk = df_chunk.drop(["year", "week"])
            
            # ⭐️ เติมชื่อ Regime ลงไปในชื่อไฟล์ด้วย!
            output_filename = os.path.join(OUTPUT_DIR, f"chunk_{regime}_{y}_W{w:02d}.parquet")
            
            df_chunk.write_parquet(output_filename)
            print(f"  -> Saved {output_filename}")

if __name__ == "__main__":
    main()