import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

# ── 4. SHAP Top 10 特征 ──────────────────────────────────────
FEATURES = [
    'systo', 'mweight', 'bl_hbalc', 'BMI', 'cesd10',
    'total_cognition', 'MS', 'gender', '消化系统疾病', 'bl_cysc'
]

TARGET = 'cmm'

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ── 5. 缺失值填充 ────────────────────────────────────────────
X = X.fillna(X.median(numeric_only=True))
y = y.dropna()
X = X.loc[y.index]

print(f"有效样本量：{len(X)}，CMM患病率：{y.mean()*100:.1f}%")

# ── 6. 归一化 ────────────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── 7. 划分训练/测试集 ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ── 8. 训练 SVM ──────────────────────────────────────────────
best_model = SVC(
    C=1, gamma='scale', kernel='rbf',
    probability=True, class_weight='balanced',
    random_state=42
)
best_model.fit(X_train, y_train)

test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"测试集 AUROC：{test_auc:.3f}")

probs = best_model.predict_proba(X_test)[:, 1]
print(f"预测概率范围：{probs.min():.3f} ~ {probs.max():.3f}，均值：{probs.mean():.3f}")

# ── 9. 保存模型 ──────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/svm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)
# 测试极端输入
test_cases = pd.DataFrame([
    {'systo': 120, 'mweight': 55, 'bl_hbalc': 5.5, 'BMI': 2,
     'cesd10': 2, 'total_cognition': 20, 'MS': 35,
     'gender': 0, '消化系统疾病': 0, 'bl_cysc': 0.7},
    {'systo': 180, 'mweight': 90, 'bl_hbalc': 10, 'BMI': 4,
     'cesd10': 28, 'total_cognition': 3, 'MS': 10,
     'gender': 1, '消化系统疾病': 1, 'bl_cysc': 2.5},
])[FEATURES]

test_scaled = scaler.transform(test_cases)
test_probs = best_model.predict_proba(test_scaled)[:, 1]
print(f"低风险测试概率：{test_probs[0]*100:.1f}%")
print(f"高风险测试概率：{test_probs[1]*100:.1f}%")

print("✅ 模型已保存至 model/ 目录")

print("✅ 模型已保存至 model/ 目录")
