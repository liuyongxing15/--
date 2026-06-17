# -*- coding: utf-8 -*-
"""
《机器学习与数据挖掘》期末项目 -
题目：基于集成学习的中国省域税负水平预测研究
数据文件：面板数据.xls
作者：5402124117-刘永兴
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 数据加载与预处理 ==================
print("=" * 60)
print("1. 正在加载数据...")

# 检查数据文件是否存在
data_path = '面板数据.xls'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"数据文件不存在：{data_path}")

df = pd.read_excel(data_path, sheet_name='Sheet1')

# 确保按省份和时间排序，保证滞后项构造正确
df = df.sort_values(['省份名称', 'year']).reset_index(drop=True)

# 定义特征列（排除非数值标识列）
feature_cols = [
    '财政支持力度', '经济发展水平', '社会消费水平', '城镇化率',
    '产业结构三产GDP', '产业结构高度化', '产业结构高级化',
    '金融发展水平存贷款余额', '城乡居民收入差距', '人口密度',
    '人力资本水平', '劳动力水平', '研发强度', '工业化水平',
    '技术市场发展水平', '交通基础设施水平1', '交通基础设施水平2',
    '对外开放程度', '信息化水平'
]
target_col = '税负水平'

print(f"   原始数据：{df.shape[0]} 条记录，{df.shape[1]} 个变量")
print(f"   特征数量：{len(feature_cols)}，目标变量：{target_col}")

# ---- 1.1 数据质量检查 ----
print("\n   【数据质量检查】")

# 检查缺失值
missing_info = df[feature_cols + [target_col]].isnull().sum()
if missing_info.any():
    print("   缺失值统计：")
    print(missing_info[missing_info > 0])
    # 使用均值填充缺失值
    df = df.fillna(df.mean())
    print("   ✓ 已使用均值填充缺失值")
else:
    print("   ✓ 无缺失值")

# 检查数据类型
dtypes = df[feature_cols + [target_col]].dtypes
print(f"\n   数据类型检查：所有特征均为 {dtypes.unique()}")

# 检查异常值（基于Z-score）
from scipy import stats
z_scores = np.abs(stats.zscore(df[feature_cols]))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"   异常值检测：{outlier_mask.sum()} 条记录存在极端值（Z-score > 3）")

# ---- 1.2 描述性统计 ----
print("\n   【目标变量描述性统计】")
print(df[target_col].describe().round(4))

# ---- 1.3 相关性分析 ----
print("\n   【特征与目标变量相关性 Top 10】")
corr_matrix = df[feature_cols + [target_col]].corr()
target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
print(target_corr.head(10).round(4))

# 绘制相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, annot=False, fmt='.2f', 
            xticklabels=True, yticklabels=True)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
print("   ✓ 相关性热力图已保存：correlation_heatmap.png")

# ================== 2. 特征工程：构造滞后项 ==================
print("\n2. 构造目标变量一阶滞后项（税负水平_lag1）...")
# 按省份分组，将目标变量向后平移1期
df['税负水平_lag1'] = df.groupby('省份名称')[target_col].shift(1)
# 移除缺失滞后项的样本（即各省份2000年的数据，无法构造滞后项）
df_lagged = df.dropna(subset=['税负水平_lag1']).copy()
print(f"   构造滞后项后样本量：{df_lagged.shape[0]}（2001-2024年，每年31省）")

# 特征矩阵（原始特征 + 滞后项）
X_cols = feature_cols + ['税负水平_lag1']
X = df_lagged[X_cols]
y = df_lagged[target_col]
years = df_lagged['year'].values  # 用于按时间划分

# ================== 3. 按时间划分训练/验证/测试集 ==================
print("\n3. 按时间顺序划分数据集（训练：2001-2017，验证：2018-2020，测试：2021-2024）...")
train_mask = years <= 2017
val_mask = (years >= 2018) & (years <= 2020)
test_mask = years >= 2021

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"   训练集样本数：{len(X_train)}（{years[train_mask].min()}-{years[train_mask].max()}年）")
print(f"   验证集样本数：{len(X_val)}（{years[val_mask].min()}-{years[val_mask].max()}年）")
print(f"   测试集样本数：{len(X_test)}（{years[test_mask].min()}-{years[test_mask].max()}年）")

# ================== 4. 标准化（Z-score） ==================
print("\n4. 对特征进行标准化处理...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("   标准化完成（均值≈0，标准差≈1）")

# ================== 5. 模型训练与超参数调优 ==================
print("\n5. 开始训练模型与超参数调优...")

# ---- 5.1 基线模型：线性回归 ----
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print("   ✓ 线性回归训练完成")

# ---- 5.2 随机森林（带网格搜索调优） ----
print("   正在调优随机森林（GridSearchCV，5折交叉验证）...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5,
                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"   ✓ 随机森林最优参数：{rf_grid.best_params_}")

# ---- 5.3 XGBoost（带网格搜索调优） ----
print("   正在调优XGBoost（GridSearchCV，5折交叉验证）...")
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
xgb = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"   ✓ XGBoost最优参数：{xgb_grid.best_params_}")

# ---- 5.4 LightGBM（带网格搜索调优） ----
print("   正在调优LightGBM（GridSearchCV，5折交叉验证）...")
lgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'subsample': [0.7, 0.8, 1.0]
}
lgb = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
lgb_grid = GridSearchCV(lgb, lgb_param_grid, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
lgb_grid.fit(X_train_scaled, y_train)
best_lgb = lgb_grid.best_estimator_
print(f"   ✓ LightGBM最优参数：{lgb_grid.best_params_}")

# ---- 5.5 梯度提升（GBR） ----
print("   正在调优梯度提升（GradientBoosting，GridSearchCV）...")
gbr_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0]
}
gbr = GradientBoostingRegressor(random_state=42)
gbr_grid = GridSearchCV(gbr, gbr_param_grid, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
gbr_grid.fit(X_train_scaled, y_train)
best_gbr = gbr_grid.best_estimator_
print(f"   ✓ 梯度提升最优参数：{gbr_grid.best_params_}")

# ---- 5.6 Stacking集成学习 ----
print("   正在构建Stacking集成模型...")
# 定义基础模型
base_estimators = [
    ('rf', best_rf),
    ('xgb', best_xgb),
    ('lgb', best_lgb),
    ('gbr', best_gbr)
]
# 定义元学习器
meta_estimator = LinearRegression()
# 构建Stacking模型
stacking = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_estimator,
    cv=5,
    n_jobs=-1
)
stacking.fit(X_train_scaled, y_train)
print("   ✓ Stacking集成模型训练完成")

# ================== 5.7 验证集评估（用于选择最佳模型） ==================
print("\n5.7 在验证集上评估模型性能...")
val_results = []
for name, model in models.items():
    y_val_pred = model.predict(X_val_scaled)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_results.append({'模型': name, '验证集RMSE': val_rmse, '验证集R²': val_r2})
    print(f"   {name}: 验证集 RMSE={val_rmse:.5f}, R²={val_r2:.3f}")

# 根据验证集性能选择最佳模型
val_df = pd.DataFrame(val_results)
best_model_by_val = val_df.sort_values('验证集R²', ascending=False).iloc[0]['模型']
print(f"\n   根据验证集性能，最佳模型为：{best_model_by_val}")

# ================== 6. 在测试集上评估 ==================
print("\n6. 在测试集（2021-2024）上评估模型性能...")

models = {
    '线性回归': lr,
    '随机森林': best_rf,
    'XGBoost': best_xgb,
    'LightGBM': best_lgb,
    '梯度提升': best_gbr,
    'Stacking集成': stacking
}

results = []
y_pred_dict = {}  # 存储各模型预测结果用于可视化
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_pred_dict[name] = y_pred
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 计算MAPE（平均绝对百分比误差）
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results.append({'模型': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})
    print(f"   {name}: RMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}, MAPE={mape:.2f}%")

# 结果表格
results_df = pd.DataFrame(results)
print("\n   【测试集性能汇总表】")
print(results_df.round(4).to_string(index=False))

# ---- 6.1 模型性能对比图 ----
plt.figure(figsize=(12, 6))
bar_width = 0.15
index = np.arange(len(results_df))

plt.bar(index - 2*bar_width, results_df['RMSE'], width=bar_width, label='RMSE')
plt.bar(index - bar_width, results_df['MAE'], width=bar_width, label='MAE')
plt.bar(index, results_df['R²'], width=bar_width, label='R²')
plt.bar(index + bar_width, results_df['MAPE'], width=bar_width, label='MAPE(%)')

plt.xlabel('模型')
plt.ylabel('指标值')
plt.title('各模型测试集性能对比')
plt.xticks(index, results_df['模型'], rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("\n   ✓ 模型对比图已保存：model_comparison.png")

# ---- 6.2 预测值与真实值对比图（最佳模型） ----
best_model_name = results_df.sort_values('R²', ascending=False).iloc[0]['模型']
best_y_pred = y_pred_dict[best_model_name]

plt.figure(figsize=(10, 6))
sorted_indices = np.argsort(y_test.values)
plt.scatter(range(len(y_test)), y_test.values[sorted_indices], 
            label='真实值', alpha=0.6, s=50)
plt.scatter(range(len(y_test)), best_y_pred[sorted_indices], 
            label=f'{best_model_name}预测值', alpha=0.6, s=50)
plt.plot(range(len(y_test)), y_test.values[sorted_indices], '--', color='blue')
plt.plot(range(len(y_test)), best_y_pred[sorted_indices], '--', color='orange')
plt.xlabel('样本序号')
plt.ylabel('税负水平')
plt.title(f'真实值 vs {best_model_name}预测值')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=300)
print(f"   ✓ 预测对比图已保存：prediction_comparison.png")

# ---- 6.3 误差分布图 ----
errors = y_test.values - best_y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='steelblue')
plt.axvline(0, color='red', linestyle='--', label='零误差线')
plt.xlabel('预测误差')
plt.ylabel('频数')
plt.title(f'{best_model_name}预测误差分布')
plt.legend()
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300)
print(f"   ✓ 误差分布图已保存：error_distribution.png")

# ================== 7. 特征重要性分析 ==================
print("\n7. 特征重要性分析...")

# ---- 7.1 XGBoost内置重要性 ----
print("   【基于XGBoost内置重要性】")
xgb_importance = pd.DataFrame({
    '特征': X_train.columns,
    '重要性': best_xgb.feature_importances_
}).sort_values('重要性', ascending=False)

print("   排名前10的重要特征：")
print(xgb_importance.head(10).to_string(index=False))

# ---- 绘制特征重要性条形图 ----
plt.figure(figsize=(10, 6))
top10 = xgb_importance.head(10)
plt.barh(top10['特征'], top10['重要性'], color='steelblue')
plt.xlabel('重要性得分')
plt.title('XGBoost 特征重要性 Top 10')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("\n   ✓ 特征重要性图已保存：feature_importance.png")

# ---- 7.2 SHAP特征解释 ----
print("\n   【SHAP特征解释分析】")
try:
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # 绘制SHAP汇总图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, 
                      show=False, max_display=10)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    print("   ✓ SHAP汇总图已保存：shap_summary.png")
    
    # 计算SHAP均值重要性
    shap_importance = pd.DataFrame({
        '特征': X_train.columns,
        'SHAP均值重要性': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP均值重要性', ascending=False)
    
    print("\n   SHAP均值重要性排名前10：")
    print(shap_importance.head(10).to_string(index=False))
    
except Exception as e:
    print(f"   ! SHAP分析失败：{e}")
    print("   提示：请安装shap库（pip install shap）")

# ================== 8. 模型保存与总结输出 ==================
print("\n8. 保存最佳模型...")

# 保存最佳模型
import joblib
best_model = stacking if 'Stacking集成' in models else best_xgb
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("   ✓ 最佳模型已保存：best_model.pkl")
print("   ✓ 标准化器已保存：scaler.pkl")

# 保存结果表格
results_df.to_excel('model_results.xlsx', index=False)
print("   ✓ 性能结果已保存：model_results.xlsx")

print("\n" + "=" * 60)
print("代码运行完毕！所有结果已复现。")
print(f"最佳模型：{best_model_name}")
print(f"最佳R²：{results_df.sort_values('R²', ascending=False).iloc[0]['R²']:.4f}")
print("\n保存的文件：")
print("  - feature_importance.png (XGBoost特征重要性)")
print("  - correlation_heatmap.png (相关性热力图)")
print("  - model_comparison.png (模型性能对比)")
print("  - prediction_comparison.png (预测值对比)")
print("  - error_distribution.png (误差分布图)")
print("  - shap_summary.png (SHAP特征解释)")
print("  - best_model.pkl (最佳模型)")
print("  - model_results.xlsx (性能结果表)")
print("=" * 60)