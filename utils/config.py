"""项目统一配置。"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "wind_speed_data.xlsx"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据列配置
SHEET_NAME = 0
TARGET_COLUMN = "Wind Speed (mph)"
# 多变量输入特征（包含目标列本身，让模型同时看历史所有气象变量）
FEATURE_COLUMNS = [
    "ETo (in)",
    "Precip (in)",
    "Sol Rad (Ly/day)",
    "Vap Pres (mBars)",
    "Air Temp (F)",
    "Rel Hum (%)",
    "Dew Point (F)",
    "Wind Speed (mph)",   # 目标列也在里面，用它的历史值做自回归
    "Wind Dir (0-360)",
    "Soil Temp (F)",
]
DATE_COLUMN = "Date"
HOUR_COLUMN = "Hour (PST)"

# 使用前 24 个连续观测值预测下一个观测值
LOOKBACK = 24

# 按时间顺序划分，禁止随机打乱，避免未来数据泄漏
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 深度学习训练配置
SEED = 42
BATCH_SIZE = 64
EPOCHS = 120
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0  # Windows 下建议保持 0

# 模型结构
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
DROPOUT = 0.10

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128

BP_HIDDEN_DIMS = (128, 64)

# 严格使用 24 阶自回归项；属于 ARIMA(p,d,q) 框架
ARIMA_ORDER = (24, 0, 0)

# 图形配置：None 表示绘制全部测试数据；数据较多时建议保留 300
PLOT_POINTS = 300
SHOW_PLOTS = True
