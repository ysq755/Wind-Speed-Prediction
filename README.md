# Wind-Speed-Prediction (风速预测)

## 📖 项目简介
本项目依托大连海洋大学信息工程学院，旨在基于多变量气象数据（风速、温度、湿度、气压等），利用时间序列预测与大模型技术，对海洋/沿海地区风速进行精准预测。

## 📁 目录结构
- `data/`: 存放原始数据与清洗后的数据 (已通过 `.gitignore` 忽略大文件)
- `Step1_data_process.py`: 数据初步探查脚本
- `Step2_clean_data.py`: 数据清洗、异常值剔除、1小时重采样脚本
- `实验日志.md`: 记录科研进展、踩坑记录与实验计划
- `models/` / `*.py`: 各种基线模型与融合模型代码 (LSTM, Transformer, Hybrid等)

## ⚙️ 环境依赖
请先安装依赖包：
`pip install -r requirements.txt`

## 🚀 快速开始
1. 将原始风速数据放置于 `data/wind_speed_data.xlsx`
2. 运行数据清洗脚本：
   `python Step2_clean_data.py`
3. 输出文件：生成的 `data/wind_speed_clean.csv` 供模型直接使用

## 📈 实验结果 (待更新)
| 模型 | MSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| LSTM | ? | ? | ? |
| Transformer | ? | ? | ? |
| Hybrid_BP_Transformer | ? | ? | ? |

## 👨‍💻 作者
- ysq755 (大连海洋大学 计算机科学与技术 硕士研究生)
