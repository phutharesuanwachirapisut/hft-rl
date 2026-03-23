import os
import polars as pl
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    print("🧠 Starting SHAP Feature Selection Analysis (L1 vs L2 Microstructure)...")
    
    # 1. โหลดข้อมูลที่ผ่าน Pipeline มาแล้ว (สมมติว่าตอนนี้มีทั้ง VPIN, OBI, OFI ครบแล้ว)
    # ถ้าคุณเซฟแยกไฟล์กัน อย่าลืม Join ข้อมูลให้มาอยู่ในไฟล์เดียวก่อนนะครับ
    DATA_PATH = "/Users/zone/Documents/Project/TradingBot/RL/data/processed/BTCUSDT_features.parquet"
    
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Warning: Could not find {DATA_PATH}.")
        print("Please ensure your dataset contains: tfi, volume, volatility_60s, vpin, obi, ofi_rolling_10")
        return
        
    df = pl.read_parquet(DATA_PATH)
    
    # 2. สร้าง Target Variable: คาดการณ์ "Log Return ในอีก 5 วินาทีข้างหน้า"
    # เราใช้ Mid-price แทน Close price เพราะ L2 Data ให้ Mid-price ที่สะท้อนตลาดได้แม่นกว่า
    print("📈 Calculating 5-second Future Returns as Target...")
    df = df.with_columns([
        # สมมติว่ามีคอลัมน์ mid_price ถ้าไม่มีใช้ price เหมือนเดิมได้ครับ
        (pl.col("price").shift(-5) / pl.col("price")).log().alias("target_return_5s")
    ]).drop_nulls()

    # 3. ⭐️ กำหนดตัวแปรต้น (Features) เข้าสู่ลานประลอง
    feature_cols = [
        "tfi",             # Trade Flow Imbalance
        "volume",          # Total Volume
        "volatility_60s",  # Market Volatility
        "vpin"            # Toxic Flow Probability
        # "obi",             # Order Book Imbalance (L2)
        # "ofi_rolling_10"   # Order Flow Imbalance (L2)
    ]
    
    # ดึง Data ออกมาเป็น Numpy (ตรวจสอบก่อนว่ามีคอลัมน์ครบ)
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
    X = df.select(feature_cols).to_numpy()
    y = df.select("target_return_5s").to_numpy().flatten()
    
    # 4. แบ่งข้อมูล Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"🚂 Training XGBoost Proxy Model on {len(X_train)} samples...")
    # 5. ฝึกสอน XGBoost แบบไวๆ
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("🔍 Calculating SHAP Values (This might take a moment)...")
    explainer = shap.TreeExplainer(model)
    
    # สุ่มดึงมา 5000 ตัวอย่างสำหรับ Plot กราฟเพื่อป้องกัน Mac RAM เต็ม
    sample_idx = np.random.choice(X_test.shape[0], 5000, replace=False)
    X_sample = X_test[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    # 6. วาดกราฟ SHAP Summary Plot
    print("📉 Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    plt.title("L1 vs L2 Microstructure SHAP Importance (Target: 5s Return)", fontsize=14, fontweight='bold')
    
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_cols, 
        show=False 
    )
    
    os.makedirs("/Users/zone/Documents/Project/TradingBot/RL/notebooks", exist_ok=True)
    plot_path = "/Users/zone/Documents/Project/TradingBot/RL/notebooks/shap_l2_importance.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ SHAP Analysis Complete! Saved chart to: {plot_path}")

if __name__ == "__main__":
    main()