rl_market_maker/
│
├── data/                       # ⚠️ โฟลเดอร์นี้ห้าม Push ลง GitHub เด็ดขาด (ใส่ใน .gitignore)
│   ├── raw/                    # เก็บไฟล์ .csv หรือ .zip ที่โหลดดิบๆ มาจาก Binance Vision
│   └── processed/              # เก็บไฟล์ .parquet ที่ผ่านการบีบอัดด้วย Polars แล้ว (ใช้เนื้อที่น้อย อ่านไว)
│
├── configs/                    # ศูนย์รวมการตั้งค่าทั้งหมด (เปลี่ยนค่าตรงนี้ โค้ดหลักไม่ต้องแก้)
│   ├── hyperparameters.yaml    # ตั้งค่าของ PPO (เช่น learning_rate, batch_size, gamma)
│   └── trading_env.yaml        # ตั้งค่ากระดานเทรด (เช่น ค่าธรรมเนียม Binance 0.1%, ค่าปรับ λ, η)
│
├── notebooks/                  # สำหรับ Jupyter Notebook (งาน R&D และพล็อตกราฟ)
│   ├── 01_data_exploration.ipynb # ส่องดูหน้าตาข้อมูล LOB และกระจายตัวของ Spread
│   └── 02_model_evaluation.ipynb # พล็อตกราฟดูผล Backtest หรือดูพฤติกรรมของ \gamma ที่ AI เลือก
│
├── src/                        # 🧠 โค้ดหลักของโปรเจกต์ (Production Code)
│   ├── __init__.py
│   ├── data_pipeline/
│   │   ├── binance_parser.py   # สคริปต์โหลดและทำความสะอาดข้อมูล (ใช้ Polars)
│   │   └── features.py         # คำนวณ State Space เช่น Volatility, OFI (Order Flow Imbalance)
│   │
│   ├── simulator/
│   │   ├── matching_engine.py  # ⚡️ โค้ดแกนกลาง LOB Simulator (เขียนด้วย Numba @njit ให้เร็วที่สุด)
│   │   └── market_env.py       # คลาส Custom Environment ที่สืบทอดจาก Gymnasium
│   │
│   ├── strategy/
│   │   ├── math_baseline.py    # สมการ Avellaneda-Stoikov คำนวณราราคาอ้างอิงและ Spread
│   │   └── rewards.py          # Custom Reward Function (คำนวณ PnL ลบด้วย Inventory Penalty)
│   │
│   └── execution/              # สำหรับต่อ API เทรดจริงในอนาคต (Phase สุดท้าย)
│       ├── binance_ws.py       # เชื่อมต่อ WebSocket ดึงสตรีม @depth แบบ Real-time (Asyncio)
│       └── order_router.py     # ระบบส่งคำสั่ง Limit Order
│
├── scripts/                    # 🚀 สคริปต์สำหรับกดรัน (Entry Points)
│   ├── 01_build_dataset.py     # สั่งแปลงไฟล์ raw -> processed (รันครั้งเดียวจบ)
│   ├── 02_train_agent.py       # สั่งเริ่มเทรนโมเดล PPO ด้วย Stable-Baselines3
│   ├── 03_run_backtest.py      # รันโมเดลที่เทรนเสร็จแล้วกับข้อมูลที่ไม่เคยเห็น (Out-of-sample)
│   └── 04_live_trading.py      # รันบอทต่อกับกระดานจริง
│
├── tests/                      # ระบบตรวจสอบความถูกต้อง (Unit Tests)
│   └── test_matching.py        # ใช้ Pytest จำลองออเดอร์วิ่งเข้า LOB ว่า PnL คำนวณเป๊ะไหม (สำคัญมาก!)
│
├── .env                        # เก็บ API Keys และ Secret ของ Binance (ห้ามแชร์!)
├── .gitignore                  # ละเว้นโฟลเดอร์ /data, /logs, .env, __pycache__
├── requirements.txt            # รายชื่อไลบรารีที่ต้อง pip install (polars, numba, stable-baselines3)
└── README.md                   # อธิบายโปรเจกต์และวิธี Setup สำหรับรันบน Mac

- เปิด Terminal พิมพ์ tensorboard --logdir ./logs/ppo_market_maker/ ระหว่างที่บอทกำลังเทรน เพื่อดูกราฟ Reward (PnL - Penalty) แบบ Real-time ได้เลย# RL-BotTrading
