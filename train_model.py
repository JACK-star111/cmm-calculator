import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import pickle
import os

# ── 1. 读取数据 ──────────────────────────────────────────────
df = pd.read_csv("ml_data.csv", encoding="utf-8-sig")

# ── 2. 分类变量编码 ──────────────────────────────────────────
df['gender'] = df['gender'].map({'男性': 0, '女性': 1})
df['cmm']    = df['cmm'].map({'不患病': 0, '患病': 1})

for col in ['消化系统疾病']:
    if df[col].dtype == object:
        df[col] = df[col].map({'否': 0, '是': 1})
    else:
        df[col] = df[col].astype(int)

# ── 3. SHAP Top 10 特征 ──────────────────────────────────────
FEATURES = [
    'systo',       # 收缩压
    'mweight',     # 体重
    'bl_hbalc',    # HbA1c
    'BMI',         # BMI
    'cesd10',      # 抑郁评分
    'total_cognition',  # 认知评分
    'MS',          # 握力
    'gender',      # 性别
    '消化系统疾病',  # 消化系统疾病
    'bl_cysc',     # 胱抑素C
]
TARGET = 'cmm'

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ── 4. 缺失值填充 ────────────────────────────────────────────
X = X.fillna(X.median(numeric_only=True))
y = y.dropna()
X = X.loc[y.index]

print(f"有效样本量：{len(X)}，CMM患病率：{y.mean()*100:.1f}%")

# ── 5. 归一化 ────────────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── 6. 划分训练/测试集 ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ── 7. 网格搜索训练 SVM ──────────────────────────────────────
param_grid = {
    'C':      [0.1, 1, 10],
    'gamma':  ['scale', 'auto'],
    'kernel': ['rbf']
}
grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced'),
    param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
test_auc   = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"最优参数：{grid.best_params_}")
print(f"测试集 AUROC：{test_auc:.3f}")

# ── 8. 保存模型 ──────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/svm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("✅ 模型已保存至 model/ 目录")
