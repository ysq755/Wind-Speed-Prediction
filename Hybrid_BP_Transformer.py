##################第一版，R^2只有0.4
# import os
# import numpy as np
# import pandas as pd
#
# import torch
# import torch.nn as nn
#
# from torch.utils.data import DataLoader, TensorDataset
#
# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     r2_score
# )
#
# import matplotlib.pyplot as plt
#
#
#
# # =====================================================
# # 路径
# # =====================================================
#
#
# BASE_DIR = r"C:\Users\ysq75\Desktop\wind_speed_forecasting"
#
#
# BP_FILE = os.path.join(
#     BASE_DIR,
#     "outputs",
#     "BP_prediction.csv"
# )
#
#
# TRANSFORMER_FILE = os.path.join(
#     BASE_DIR,
#     "outputs",
#     "Transformer_prediction.csv"
# )
#
#
# OUTPUT_DIR = os.path.join(
#     BASE_DIR,
#     "outputs"
# )
#
#
#
# SEQ_LEN = 24
#
# EPOCHS = 150
#
# LR = 0.001
#
# BATCH_SIZE = 16
#
#
#
# # =====================================================
# # 读取预测结果
# # =====================================================
#
#
# def load_prediction_data():
#
#
#     print("\n读取BP预测文件")
#
#     bp=pd.read_csv(
#         BP_FILE
#     )
#
#
#     print(bp.columns)
#
#
#
#     print("\n读取Transformer预测文件")
#
#
#     tf=pd.read_csv(
#         TRANSFORMER_FILE
#     )
#
#
#     print(tf.columns)
#
#
#
#     # 统一字段
#
#
#     y_true = bp[
#         "actual"
#     ].values
#
#
#     bp_pred = bp[
#         "prediction"
#     ].values
#
#
#
#     tf_pred = tf[
#         "prediction"
#     ].values
#
#
#
#     print("\n数据长度")
#
#     print(
#         "Actual:",
#         len(y_true)
#     )
#
#     print(
#         "BP:",
#         len(bp_pred)
#     )
#
#     print(
#         "Transformer:",
#         len(tf_pred)
#     )
#
#
#     return y_true,bp_pred,tf_pred
#
#
#
#
#
# # =====================================================
# # 构造Hybrid输入
# # =====================================================
#
#
# def create_dataset(
#         bp,
#         transformer,
#         y
# ):
#
#
#     X=[]
#
#     Y=[]
#
#
#
#     for i in range(len(y)-SEQ_LEN):
#
#
#         feature=np.column_stack(
#
#             (
#                 bp[i:i+SEQ_LEN],
#                 transformer[i:i+SEQ_LEN]
#             )
#
#         )
#
#
#         X.append(feature)
#
#
#         Y.append(
#             y[i+SEQ_LEN]
#         )
#
#
#
#     return np.array(X),np.array(Y)
#
#
#
#
#
# # =====================================================
# # Hybrid模型
# # =====================================================
#
#
# class HybridBPTransformer(nn.Module):
#
#
#     def __init__(self):
#
#         super().__init__()
#
#
#         # BP分支
#
#         self.bp_branch=nn.Sequential(
#
#             nn.Linear(
#                 SEQ_LEN*2,
#                 64
#             ),
#
#             nn.ReLU(),
#
#
#             nn.Linear(
#                 64,
#                 32
#             ),
#
#             nn.ReLU()
#
#         )
#
#
#
#         # Transformer分支
#
#
#         self.input_fc=nn.Linear(
#             2,
#             32
#         )
#
#
#
#         encoder_layer=nn.TransformerEncoderLayer(
#
#             d_model=32,
#
#             nhead=4,
#
#             batch_first=True
#
#         )
#
#
#         self.transformer=nn.TransformerEncoder(
#
#             encoder_layer,
#
#             num_layers=2
#
#         )
#
#
#         self.fc_trans=nn.Linear(
#             32,
#             32
#         )
#
#
#
#         # 融合层
#
#
#         self.output=nn.Sequential(
#
#             nn.Linear(
#                 64,
#                 32
#             ),
#
#             nn.ReLU(),
#
#
#             nn.Linear(
#                 32,
#                 1
#             )
#
#         )
#
#
#
#     def forward(self,x):
#
#
#         # BP
#
#         bp_input=x.reshape(
#             x.size(0),
#             -1
#         )
#
#
#         bp_feature=self.bp_branch(
#             bp_input
#         )
#
#
#
#         # Transformer
#
#
#         trans=self.input_fc(
#             x
#         )
#
#
#         trans=self.transformer(
#             trans
#         )
#
#
#         trans=torch.mean(
#             trans,
#             dim=1
#         )
#
#
#         trans_feature=self.fc_trans(
#             trans
#         )
#
#
#
#         # fusion
#
#
#         feature=torch.cat(
#
#             [
#                 bp_feature,
#                 trans_feature
#             ],
#
#             dim=1
#
#         )
#
#
#         out=self.output(
#             feature
#         )
#
#
#         return out
#
#
#
#
#
# # =====================================================
# # 主程序
# # =====================================================
#
#
# def main():
#
#
#
#     y_true,bp_pred,tf_pred = load_prediction_data()
#
#
#
#     X,y=create_dataset(
#
#         bp_pred,
#
#         tf_pred,
#
#         y_true
#
#     )
#
#
#
#     print("\n训练数据")
#
#     print(
#         X.shape,
#         y.shape
#     )
#
#
#
#     train_size=int(
#         len(X)*0.8
#     )
#
#
#
#     X_train=X[:train_size]
#
#     y_train=y[:train_size]
#
#
#     X_test=X[train_size:]
#
#     y_test=y[train_size:]
#
#
#
#     train_loader=DataLoader(
#
#         TensorDataset(
#
#             torch.FloatTensor(X_train),
#
#             torch.FloatTensor(y_train)
#
#         ),
#
#         batch_size=BATCH_SIZE,
#
#         shuffle=True
#
#     )
#
#
#
#     model=HybridBPTransformer()
#
#
#
#     optimizer=torch.optim.Adam(
#
#         model.parameters(),
#
#         lr=LR
#
#     )
#
#
#
#     loss_fn=nn.MSELoss()
#
#
#
#     print("\n开始训练 Hybrid BP-Transformer")
#
#
#
#     for epoch in range(EPOCHS):
#
#
#         model.train()
#
#
#         total_loss=0
#
#
#         for bx,by in train_loader:
#
#
#             pred=model(bx)
#
#
#             loss=loss_fn(
#
#                 pred,
#
#                 by.reshape(-1,1)
#
#             )
#
#
#             optimizer.zero_grad()
#
#
#             loss.backward()
#
#
#             optimizer.step()
#
#
#
#             total_loss+=loss.item()
#
#
#
#         if (epoch+1)%20==0:
#
#
#             print(
#
#                 f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss:.6f}"
#
#             )
#
#
#
#
#
#     # ================================
#     # 测试
#     # ================================
#
#
#     model.eval()
#
#
#     with torch.no_grad():
#
#
#         pred=model(
#
#             torch.FloatTensor(X_test)
#
#         ).numpy()
#
#
#
#     pred=pred.flatten()
#
#
#
#     print("\n====================")
#
#     print("Hybrid BP-Transformer Result")
#
#     print("====================")
#
#
#
#     rmse=np.sqrt(
#
#         mean_squared_error(
#
#             y_test,
#
#             pred
#
#         )
#
#     )
#
#
#     mae=mean_absolute_error(
#
#         y_test,
#
#         pred
#
#     )
#
#
#     mape=np.mean(
#
#         np.abs(
#
#             (y_test-pred)/y_test
#
#         )
#
#     )*100
#
#
#
#     r2=r2_score(
#
#         y_test,
#
#         pred
#
#     )
#
#
#
#     print(
#         f"RMSE:{rmse:.6f}"
#     )
#
#     print(
#         f"MAE:{mae:.6f}"
#     )
#
#     print(
#         f"MAPE:{mape:.4f}%"
#     )
#
#     print(
#         f"R2:{r2:.6f}"
#     )
#
#
#
#
#     # 保存结果
#
#
#     result=pd.DataFrame({
#
#         "Actual":
#
#         y_test,
#
#
#         "Hybrid_BP_Transformer":
#
#         pred
#
#     })
#
#
#
#     result.to_csv(
#
#         os.path.join(
#
#             OUTPUT_DIR,
#
#             "Hybrid_BP_Transformer_predictions.csv"
#
#         ),
#
#         index=False
#
#     )
#
#
#
#     metrics=pd.DataFrame({
#
#         "Model":
#
#         [
#             "Hybrid BP-Transformer"
#         ],
#
#
#         "RMSE":
#
#         [
#             rmse
#         ],
#
#
#         "MAE":
#
#         [
#             mae
#         ],
#
#
#         "MAPE":
#
#         [
#             mape
#         ],
#
#
#         "R2":
#
#         [
#             r2
#         ]
#
#     })
#
#
#     metrics.to_csv(
#
#         os.path.join(
#
#             OUTPUT_DIR,
#
#             "Hybrid_BP_Transformer_metrics.csv"
#
#         ),
#
#         index=False
#
#     )
#
#
#
#
#
#     # 绘图
#
#
#     plt.figure(
#
#         figsize=(10,5)
#
#     )
#
#
#     plt.plot(
#
#         y_test,
#
#         label="Actual"
#
#     )
#
#
#     plt.plot(
#
#         pred,
#
#         label="Hybrid BP-Transformer"
#
#     )
#
#
#     plt.xlabel(
#         "Sample"
#     )
#
#
#     plt.ylabel(
#         "Wind Speed"
#     )
#
#
#     plt.title(
#
#         "Hybrid BP-Transformer Prediction"
#
#     )
#
#
#     plt.legend()
#
#
#     plt.grid(alpha=0.3)
#
#
#     plt.tight_layout()
#
#
#     plt.savefig(
#
#         os.path.join(
#
#             OUTPUT_DIR,
#
#             "Hybrid_BP_Transformer.png"
#
#         ),
#
#         dpi=300
#
#     )
#
#
#     print("\n全部完成")
#
#
#
#
#
# if __name__=="__main__":
#
#     main()


'''第二版，R^2=0.56'''
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


BASE_DIR = r"C:\Users\ysq75\Desktop\wind_speed_forecasting"
BP_FILE = os.path.join(BASE_DIR, "outputs", "BP_prediction.csv")
TF_FILE = os.path.join(BASE_DIR, "outputs", "Transformer_prediction.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

VAL_RATIO = 0.3   # 用后 30% 当验证集搜权重，剩下 70% 当测试集评估


def main():
    bp = pd.read_csv(BP_FILE)
    tf = pd.read_csv(TF_FILE)

    n = min(len(bp), len(tf))
    bp = bp.iloc[:n].reset_index(drop=True)
    tf = tf.iloc[:n].reset_index(drop=True)

    y_true = bp["actual"].values
    bp_pred = bp["prediction"].values
    tf_pred = tf["prediction"].values

    # 划分验证集（后 30%）和测试集（前 70%）
    split = int(n * (1 - VAL_RATIO))

    y_val = y_true[split:]
    bp_val = bp_pred[split:]
    tf_val = tf_pred[split:]

    y_test = y_true[:split]
    bp_test = bp_pred[:split]
    tf_test = tf_pred[:split]

    # ---- 在验证集上网格搜索最优权重 w ----
    best_w, best_mse = 0.5, float("inf")
    for w in np.linspace(0, 1, 101):
        pred_val = w * bp_val + (1 - w) * tf_val
        mse = mean_squared_error(y_val, pred_val)
        if mse < best_mse:
            best_mse = mse
            best_w = w

    print(f"\n最优权重: BP={best_w:.2f}, Transformer={1-best_w:.2f}")
    print(f"验证集 MSE: {best_mse:.6f}")

    # ---- 在测试集上评估 ----
    pred_test = best_w * bp_test + (1 - best_w) * tf_test

    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    eps = 1e-8
    mape = np.mean(np.abs((y_test - pred_test) / (np.abs(y_test) + eps))) * 100
    r2 = r2_score(y_test, pred_test)

    print("\n====================")
    print("Hybrid Weighted Ensemble (BP + Transformer)")
    print("====================")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2  : {r2:.6f}")

    # 保存
    result = pd.DataFrame({
        "Actual": y_test,
        "Weighted_Ensemble": pred_test,
        "BP": bp_test,
        "Transformer": tf_test
    })
    result.to_csv(
        os.path.join(OUTPUT_DIR, "Hybrid_Weighted_Ensemble_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    metrics = pd.DataFrame({
        "Model": ["Hybrid Weighted Ensemble"],
        "BP_Weight": [best_w],
        "TF_Weight": [1 - best_w],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE": [mape],
        "R2": [r2]
    })
    metrics.to_csv(
        os.path.join(OUTPUT_DIR, "Hybrid_Weighted_Ensemble_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual")
    plt.plot(pred_test, label=f"Ensemble (w={best_w:.2f})")
    plt.plot(bp_test, label="BP", alpha=0.4)
    plt.plot(tf_test, label="Transformer", alpha=0.4)
    plt.xlabel("Sample")
    plt.ylabel("Wind Speed")
    plt.title("Hybrid Weighted Ensemble: BP + Transformer")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Hybrid_Weighted_Ensemble.png"),
        dpi=300
    )

    print("\n全部完成")


if __name__ == "__main__":
    main()