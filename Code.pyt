# =============================================================================
# BLOCK 1: IMPORTS AND SETUP
# =============================================================================
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully.")


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Run risk clustering and supervised modeling.')
    parser.add_argument('--dataset', default='Dataset2.csv',
                        help='Path to the input CSV dataset file (default: Dataset2.csv)')
    return parser.parse_args()


args = parse_args()
print(f"Loading dataset from: {args.dataset}")

# =============================================================================
# BLOCK 2: LOAD AND INSPECT
# =============================================================================
df = load_dataset(args.dataset)

print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))

orig_rows = df.shape[0]
df = df.drop_duplicates()
print(f"\nDuplicates removed: {orig_rows - df.shape[0]}, rows left: {df.shape[0]}")

# =============================================================================
# BLOCK 3: ENCODING
#
# DESIGN RATIONALE:
#   - Every categorical column is encoded as a SINGLE ordinal score
#     ordered from least risky / conservative to most risky / aggressive.
#   - This avoids the one-hot problem where e.g. "Objective_Growth" exists
#     without "Objective_Income", making the supervised model see a fragment
#     of a column rather than the full answer.
#   - Source and Investment_Avenues are dropped: Source has no risk meaning,
#     Investment_Avenues is 92% "Yes" (near-zero variance, adds no signal).
#   - Rank columns: original scale is 1=most preferred, 7=least preferred.
#     We invert (8 - rank) so higher score = stronger preference.
# =============================================================================

rank_cols = ['Mutual_Funds', 'Equity_Market', 'Debentures',
             'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
for col in rank_cols:
    df[col] = 8 - pd.to_numeric(df[col], errors='coerce')

# Binary
df['Gender_Score'] = df['gender'].map({'Female': 0, 'Male': 1})
df['Stock_Score']  = df['Stock_Marktet'].map({'No': 0, 'Yes': 1})

# Ordinal: stated preferences (ordered conservative → aggressive)
df['Duration_Score']    = df['Duration'].map(
    {'Less than 1 year': 1, '1-3 years': 2, '3-5 years': 3, 'More than 5 years': 4})
df['Expect_Score']      = df['Expect'].map(
    {'10%-20%': 1, '20%-30%': 2, '30%-40%': 3})
df['Monitor_Score']     = df['Invest_Monitor'].map(
    {'Monthly': 1, 'Weekly': 2, 'Daily': 3})
df['Factor_Score']      = df['Factor'].map(
    {'Locking Period': 1, 'Returns': 2, 'Risk': 3})
df['Objective_Score']   = df['Objective'].map(
    {'Income': 1, 'Capital Appreciation': 2, 'Growth': 3})
df['Purpose_Score']     = df['Purpose'].map(
    {'Savings for Future': 1, 'Returns': 2, 'Wealth Creation': 3})
df['Savings_Score']     = df['What are your savings objectives?'].map(
    {'Education': 1, 'Health Care': 2, 'Retirement Plan': 3})
df['Avenue_Score']      = df['Avenue'].map(
    {'Public Provident Fund': 1, 'Fixed Deposits': 2, 'Mutual Fund': 3, 'Equity': 4})

# Ordinal: reasons for each instrument (ordered by risk-seeking behavior)
df['ReasonEq_Score']    = df['Reason_Equity'].map(
    {'Dividend': 1, 'Liquidity': 2, 'Capital Appreciation': 3})
df['ReasonMut_Score']   = df['Reason_Mutual'].map(
    {'Tax Benefits': 1, 'Fund Diversification': 2, 'Better Returns': 3})
df['ReasonBonds_Score'] = df['Reason_Bonds'].map(
    {'Tax Incentives': 1, 'Safe Investment': 2, 'Assured Returns': 3})
df['ReasonFD_Score']    = df['Reason_FD'].map(
    {'High Interest Rates': 1, 'Risk Free': 2, 'Fixed Returns': 3})

print("\nEncoding complete. Missing values per encoded column:")
encoded_cols = (rank_cols +
                ['Gender_Score','Stock_Score','Duration_Score','Expect_Score',
                 'Monitor_Score','Factor_Score','Objective_Score','Purpose_Score',
                 'Savings_Score','Avenue_Score',
                 'ReasonEq_Score','ReasonMut_Score','ReasonBonds_Score','ReasonFD_Score'])
print(df[encoded_cols].isnull().sum())

# =============================================================================
# BLOCK 4: FEATURE SETS
#
# TWO STRICTLY SEPARATED FEATURE SETS:
#
#   CLUSTERING features — actual revealed investment behavior:
#     rank columns (what investors choose) + reason scores (why they choose it).
#     These directly measure risk behavior and drive the W-KMeans labels.
#
#   SUPERVISED features — demographic and stated preference:
#     age, gender, stock market participation, investment goals, duration,
#     expected return, monitoring frequency, and avenue preference.
#     These do NOT overlap with the clustering features, preventing leakage.
#     The RF learns to predict risk class from who the investor IS and what
#     they intend — useful for classifying new investors before knowing portfolio.
#
# DROPPED columns:
#   Source         — information source, no risk meaning
#   Investment_Avenues — 92% "Yes", near-zero variance
#   Stock_Marktet  — already encoded as Stock_Score
#   gender, Objective, Purpose, etc. — already encoded above
# =============================================================================

CLUSTER_FEATURES = rank_cols + [
    'ReasonEq_Score', 'ReasonMut_Score', 'ReasonBonds_Score', 'ReasonFD_Score'
]

SUPERVISED_FEATURES = [
    'age', 'Gender_Score', 'Stock_Score',
    'Duration_Score', 'Expect_Score', 'Monitor_Score',
    'Factor_Score', 'Objective_Score', 'Purpose_Score',
    'Savings_Score', 'Avenue_Score'
]

print(f"\nClustering features ({len(CLUSTER_FEATURES)}): {CLUSTER_FEATURES}")
print(f"Supervised features ({len(SUPERVISED_FEATURES)}): {SUPERVISED_FEATURES}")

# =============================================================================
# BLOCK 5: CORRELATION CHECK ON CLUSTERING FEATURES
# =============================================================================
corr_clust = df[CLUSTER_FEATURES].corr().round(2)
plt.figure(figsize=(12, 9))
sns.heatmap(corr_clust, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, annot_kws={"size": 8})
plt.title("Correlation Matrix — Clustering Features")
plt.tight_layout()
plt.show()

THRESH = 0.85
drop_set = set()
cols = corr_clust.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if abs(corr_clust.iloc[i, j]) >= THRESH:
            drop_set.add(cols[j])

if drop_set:
    print(f"\nDropping {len(drop_set)} highly correlated clustering features: {drop_set}")
    CLUSTER_FEATURES = [c for c in CLUSTER_FEATURES if c not in drop_set]
    rank_cols = [c for c in rank_cols if c not in drop_set]
else:
    print("\nNo clustering features exceeded the correlation threshold.")

# =============================================================================
# BLOCK 6: W-KMEANS CLUSTERING
# =============================================================================
MAX_WEIGHT_CAP = 0.15

def cap_and_redistribute(weights, cap=MAX_WEIGHT_CAP):
    w = weights.copy()
    for _ in range(100):
        over = w > cap
        if not over.any(): break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.any():
            w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()

def compute_feature_wcss(X, labels, n_clusters):
    wcss = np.zeros(X.shape[1])
    for k in range(n_clusters):
        pts = X[labels == k]
        if len(pts) > 0:
            wcss += ((pts - pts.mean(axis=0)) ** 2).sum(axis=0)
    return wcss

def wkmeans(X, n_clusters=3, max_iter=50, tol=1e-4, random_state=42, beta=2.0, cap=MAX_WEIGHT_CAP):
    n, m = X.shape
    w = np.ones(m) / m
    prev_w = np.zeros(m)
    for it in range(max_iter):
        Xw = X * np.sqrt(w)
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(Xw)
        wcss = compute_feature_wcss(X, labels, n_clusters)
        new_w = np.zeros(m)
        for j in range(m):
            if wcss[j] == 0:
                new_w[j] = 1.0
            else:
                denom = np.sum((wcss[j] / (wcss + 1e-10)) ** (1.0 / (beta - 1)))
                new_w[j] = 1.0 / (denom + 1e-10)
        new_w = new_w / new_w.sum()
        new_w = cap_and_redistribute(new_w, cap=cap)
        if np.max(np.abs(new_w - prev_w)) < tol:
            print(f"  W-KMeans converged at iteration {it + 1}")
            break
        prev_w, w = new_w.copy(), new_w
    else:
        print("  W-KMeans: max iterations reached")
    return labels, w

scaler_clust = StandardScaler()
X_clust = scaler_clust.fit_transform(df[CLUSTER_FEATURES])

print("\nRunning W-KMeans...")
wk_labels, weights = wkmeans(X_clust, n_clusters=3)

wdf = pd.DataFrame({'Feature': CLUSTER_FEATURES, 'Weight': weights}).sort_values('Weight', ascending=False)
print("\nFeature weights:")
print(wdf.to_string(index=False))

plt.figure(figsize=(12, 4))
plt.bar(wdf['Feature'], wdf['Weight'], color='steelblue')
plt.axhline(y=MAX_WEIGHT_CAP, color='red', linestyle='--', label=f'Cap={MAX_WEIGHT_CAP}')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title("W-KMeans Feature Weights")
plt.legend()
plt.tight_layout()
plt.show()

df['WK_Cluster'] = wk_labels

# =============================================================================
# BLOCK 7: RISK LABEL ASSIGNMENT
# =============================================================================
risk_weights = {
    'Equity_Market':    2.0,
    'Mutual_Funds':     1.0,
    'Gold':             0.5,
    'Debentures':       0.0,
    'Government_Bonds': -1.0,
    'Fixed_Deposits':   -1.0,
    'PPF':              -0.5,
}

avail_rank = [c for c in rank_cols if c in df.columns]
profile_cols = avail_rank + ['ReasonEq_Score', 'ReasonMut_Score', 'ReasonBonds_Score', 'ReasonFD_Score']
raw_profiles = df.groupby('WK_Cluster')[avail_rank].mean().round(2)

risk_scores = {}
for cid in raw_profiles.index:
    score = sum(risk_weights.get(a, 0) * raw_profiles.loc[cid, a]
                for a in risk_weights if a in raw_profiles.columns)
    risk_scores[cid] = score

sorted_risk = sorted(risk_scores, key=risk_scores.get)
cid_con, cid_mod, cid_agg = sorted_risk
label_map = {cid_con: 0, cid_mod: 1, cid_agg: 2}
df['Risk_Class'] = df['WK_Cluster'].map(label_map)

label_names = {0: 'Conservative', 1: 'Moderate', 2: 'Aggressive'}
palette = {0: '#2196F3', 1: '#FF9800', 2: '#E53935'}
order_ids = [0, 1, 2]
order_labels = [label_names[i] for i in order_ids]
named_pal = {label_names[i]: palette[i] for i in order_ids}

print("\n=== FINAL RISK CLASS SIZES ===")
for lbl in order_ids:
    cnt = (df['Risk_Class'] == lbl).sum()
    print(f"  {label_names[lbl]:12}: {cnt:5} ({cnt / len(df) * 100:.1f}%)")

print("\nRisk scores per cluster (monotonicity check):")
for lbl in order_ids:
    orig_cid = [k for k, v in label_map.items() if v == lbl][0]
    print(f"  {label_names[lbl]:12}: {risk_scores[orig_cid]:.3f}")

# =============================================================================
# BLOCK 8: CLUSTERING VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("W-KMEANS CLUSTER QUALITY EVALUATION")
print("=" * 60)

sil = silhouette_score(X_clust, wk_labels)
db  = davies_bouldin_score(X_clust, wk_labels)
ch  = calinski_harabasz_score(X_clust, wk_labels)
print(f"\nInternal metrics:")
print(f"  Silhouette score   : {sil:.4f}  (>0.25 acceptable for survey data; >0.4 good)")
print(f"  Davies-Bouldin idx : {db:.4f}  (lower is better)")
print(f"  Calinski-Harabasz  : {ch:.1f}  (higher is better)")

print(f"\nANOVA: separation across risk classes")
print(f"{'Feature':<25} {'F-stat':>10} {'p-value':>12} {'Significant?':>14}")
print("-" * 65)
for feat in CLUSTER_FEATURES:
    groups = [df[df['Risk_Class'] == k][feat].dropna().values for k in [0, 1, 2]]
    f_stat, p_val = f_oneway(*groups)
    sig = "YES ✓" if p_val < 0.05 else "no"
    print(f"{feat:<25} {f_stat:>10.2f} {p_val:>12.4f} {sig:>14}")

# Heatmap of cluster profiles
final_profiles = df.groupby('Risk_Class')[avail_rank].mean().round(2)
final_profiles.index = [label_names[i] for i in final_profiles.index]
plt.figure(figsize=(10, 4))
sns.heatmap(final_profiles.T, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Cluster Profiles — Mean Investment Avenue Preference per Risk Class")
plt.tight_layout()
plt.show()

# Elbow + silhouette vs k
X_w = X_clust * np.sqrt(weights)
wcss_list, sil_list = [], []
k_range = range(2, 8)
for k in k_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_w)
    wcss_list.append(km_tmp.inertia_)
    sil_list.append(silhouette_score(X_w, km_tmp.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_range, wcss_list, 'o-', color='steelblue')
ax1.axvline(x=3, color='red', linestyle='--', label='k=3 chosen')
ax1.set_title("Elbow curve"); ax1.set_xlabel("k"); ax1.set_ylabel("WCSS"); ax1.legend()
ax2.plot(k_range, sil_list, 's-', color='darkorange')
ax2.axvline(x=3, color='red', linestyle='--', label='k=3 chosen')
ax2.set_title("Silhouette vs k"); ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.legend()
plt.tight_layout(); plt.show()

# =============================================================================
# BLOCK 9: DEMOGRAPHIC ANALYSIS
# =============================================================================
demo = df[['Risk_Class']].copy()
demo['Risk_Label'] = demo['Risk_Class'].map(label_names)
demo['age'] = pd.to_numeric(df['age'], errors='coerce')
demo['gender'] = df['gender'].astype(str).str.strip()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for lbl in order_ids:
    sub = demo[demo['Risk_Class'] == lbl]['age'].dropna()
    axs[0].hist(sub, bins=15, alpha=0.5, color=palette[lbl], label=label_names[lbl], edgecolor='black')
axs[0].set_title("Age Distribution by Risk Class"); axs[0].legend()
sns.boxplot(data=demo, x='Risk_Label', y='age', order=order_labels, palette=named_pal, ax=axs[1])
axs[1].set_title("Age Box Plot")
mean_age = demo.groupby('Risk_Label')['age'].mean().reindex(order_labels)
std_age  = demo.groupby('Risk_Label')['age'].std().reindex(order_labels)
axs[2].bar(order_labels, mean_age, yerr=std_age, color=[palette[i] for i in order_ids], capsize=6)
axs[2].set_title("Mean Age ± Std")
plt.suptitle("Age vs Risk Class"); plt.tight_layout(); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
gender_cnt = demo.groupby(['Risk_Label', 'gender']).size().unstack(fill_value=0).reindex(order_labels)
gender_cnt.plot(kind='bar', ax=axs[0], edgecolor='black', color=['#F48FB1', '#90CAF9'])
axs[0].set_title("Gender Count per Risk Class"); axs[0].set_ylabel("Count")
gender_pct = gender_cnt.div(gender_cnt.sum(axis=1), axis=0) * 100
gender_pct.plot(kind='bar', stacked=True, ax=axs[1], edgecolor='black', color=['#F48FB1', '#90CAF9'])
axs[1].set_title("Gender Proportion (%)"); axs[1].set_ylabel("%")
plt.suptitle("Gender vs Risk Class"); plt.tight_layout(); plt.show()

# =============================================================================
# BLOCK 10: INVESTMENT AVENUE VISUALISATIONS
# =============================================================================
n_feat = len(avail_rank)
n_cols = 3
n_rows = (n_feat + n_cols - 1) // n_cols
df_plot = df[avail_rank + ['Risk_Class']].copy()
df_plot['Risk_Label'] = df_plot['Risk_Class'].map(label_names)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), constrained_layout=True)
axes = axes.flatten()
for idx, feat in enumerate(avail_rank):
    sns.boxplot(data=df_plot, x='Risk_Label', y=feat, order=order_labels,
                palette=named_pal, ax=axes[idx])
    axes[idx].set_title(feat)
for idx in range(n_feat, len(axes)): axes[idx].set_visible(False)
fig.suptitle("Box plots: Investment avenue preference per risk class")
plt.show()

means = df.groupby('Risk_Class')[avail_rank].mean().round(2)
means.index = [label_names[i] for i in means.index]
x = np.arange(len(avail_rank)); width = 0.25
fig4, ax4 = plt.subplots(figsize=(14, 5))
for i, lbl in enumerate(order_ids):
    ax4.bar(x + i*width, means.loc[label_names[lbl]], width,
            label=label_names[lbl], color=palette[lbl], edgecolor='black', alpha=0.85)
ax4.set_xticks(x + width)
ax4.set_xticklabels(avail_rank, rotation=45, ha='right')
ax4.set_ylabel("Mean Value"); ax4.set_title("Mean Investment Avenue Preference per Risk Class")
ax4.legend(); plt.tight_layout(); plt.show()

# =============================================================================
# BLOCK 11: SUPERVISED MODEL — PREPARE DATA
#
# The supervised model trains on SUPERVISED_FEATURES only (no rank cols,
# no reason cols). This means the model learns to predict risk class from
# demographics and stated investment intent — not from the portfolio itself.
# This is intentional: it allows classifying new investors before knowing
# their actual investment allocations.
# =============================================================================
X = df[SUPERVISED_FEATURES].values.astype(float)
y = df['Risk_Class'].values

print("\n=== CLASS DISTRIBUTION ===")
for lbl in order_ids:
    cnt = (y == lbl).sum()
    print(f"  {label_names[lbl]:12}: {cnt:5} ({cnt/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train_arr = X_train.astype(float)
X_test_arr  = X_test.astype(float)
y_train_np  = y_train.flatten()
y_test_np   = y_test.flatten()
target_names = [label_names[i] for i in order_ids]
print(f"\nTrain: {X_train_arr.shape[0]}, Test: {X_test_arr.shape[0]}")

# =============================================================================
# BLOCK 12: SMOTE AND UNDERSAMPLING VARIANTS
# =============================================================================
sm = SMOTE(random_state=42, k_neighbors=5)
X_sm, y_sm = sm.fit_resample(X_train_arr, y_train_np)
y_sm = y_sm.flatten()
print(f"SMOTE training size: {X_sm.shape[0]}")

rus = RandomUnderSampler(random_state=42)
X_us, y_us = rus.fit_resample(X_train_arr, y_train_np)
y_us = y_us.flatten()
print(f"Undersampled training size: {X_us.shape[0]}")

# =============================================================================
# BLOCK 13: EVALUATION FUNCTION
# =============================================================================
def evaluate_phase(name, X_tr, y_tr, use_class_weight=False, cmap='Blues'):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"{'Model':<18} | {'Accuracy':>10} | {'Balanced Acc':>13} | {'Macro F1':>10}")
    print("-" * 60)
    results = {}

    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42,
                                class_weight='balanced' if use_class_weight else None)
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_test_arr).flatten()
    acc, bal, f1 = accuracy_score(y_test_np, yp), balanced_accuracy_score(y_test_np, yp), f1_score(y_test_np, yp, average='macro')
    print(f"{'Random Forest':<18} | {acc:>10.4f} | {bal:>13.4f} | {f1:>10.4f}")
    results['Random Forest'] = {'model': rf, 'y_pred': yp, 'acc': acc, 'bal_acc': bal, 'f1': f1}

    xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                        eval_metric='mlogloss', use_label_encoder=False)
    sw = compute_sample_weight('balanced', y=y_tr) if use_class_weight else None
    xgb.fit(X_tr, y_tr, sample_weight=sw)
    yp = xgb.predict(X_test_arr).flatten()
    acc, bal, f1 = accuracy_score(y_test_np, yp), balanced_accuracy_score(y_test_np, yp), f1_score(y_test_np, yp, average='macro')
    print(f"{'XGBoost':<18} | {acc:>10.4f} | {bal:>13.4f} | {f1:>10.4f}")
    results['XGBoost'] = {'model': xgb, 'y_pred': yp, 'acc': acc, 'bal_acc': bal, 'f1': f1}

    cb = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_state=42)
    cb.fit(X_tr, y_tr, sample_weight=sw)
    yp = cb.predict(X_test_arr).flatten()
    acc, bal, f1 = accuracy_score(y_test_np, yp), balanced_accuracy_score(y_test_np, yp), f1_score(y_test_np, yp, average='macro')
    print(f"{'CatBoost':<18} | {acc:>10.4f} | {bal:>13.4f} | {f1:>10.4f}")
    results['CatBoost'] = {'model': cb, 'y_pred': yp, 'acc': acc, 'bal_acc': bal, 'f1': f1}

    for model_name, res in results.items():
        print(f"\n--- {model_name} classification report ---")
        print(classification_report(y_test_np, res['y_pred'], target_names=target_names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (model_name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test_np, res['y_pred'], labels=order_ids)
        sns.heatmap(pd.DataFrame(cm,
                    index=[f"True:{label_names[i]}" for i in order_ids],
                    columns=[f"Pred:{label_names[i]}" for i in order_ids]),
                    annot=True, fmt='d', cmap=cmap, ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
        ax.set_title(f"{model_name}\n({name})", fontsize=11, fontweight='bold')
    plt.suptitle(f"Confusion matrices – {name}", fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()
    return results

# =============================================================================
# BLOCK 14: FOUR-WAY BALANCING COMPARISON
# =============================================================================
phase1 = evaluate_phase("PHASE 1: NO BALANCING",       X_train_arr, y_train_np, use_class_weight=False, cmap='Greys')
phase2 = evaluate_phase("PHASE 2: CLASS WEIGHTING",    X_train_arr, y_train_np, use_class_weight=True,  cmap='Blues')
phase3 = evaluate_phase("PHASE 3: SMOTE",              X_sm,        y_sm,       use_class_weight=False, cmap='Greens')
phase4 = evaluate_phase("PHASE 4: UNDERSAMPLING",      X_us,        y_us,       use_class_weight=False, cmap='Oranges')

print("\n" + "=" * 70)
print("FOUR-WAY COMPARISON (EXCLUDING NO-BALANCING)")
print("=" * 70)
for model in ['Random Forest', 'XGBoost', 'CatBoost']:
    print(f"\n{model}")
    print(f"{'Metric':<15} {'CW Only':>10} {'SMOTE':>10} {'Undersamp':>12}")
    print("-" * 47)
    for metric, key in [('Accuracy', 'acc'), ('Balanced Acc', 'bal_acc'), ('Macro F1', 'f1')]:
        print(f"{metric:<15} {phase2[model][key]:>10.4f} "
              f"{phase3[model][key]:>10.4f} {phase4[model][key]:>12.4f}")

# =============================================================================
# BLOCK 15: SELECT BEST BALANCED MODEL AND CROSS-VALIDATION
# =============================================================================
from sklearn.metrics import make_scorer

print("\n" + "=" * 70)
print("SELECTING BEST MODEL (AMONG BALANCED APPROACHES)")
print("=" * 70)

# Compare balanced accuracy scores among balanced phases only
best_score = -1
best_model_name = None
best_phase_name = None
best_phase = None
best_X_train = None
best_y_train = None

for phase_name, phase_dict, X_tr, y_tr in [
    ("CLASS WEIGHTING", phase2, X_train_arr, y_train_np),
    ("SMOTE", phase3, X_sm, y_sm),
    ("UNDERSAMPLING", phase4, X_us, y_us)
]:
    for model_name in ['Random Forest', 'XGBoost', 'CatBoost']:
        bal_acc = phase_dict[model_name]['bal_acc']
        if bal_acc > best_score:
            best_score = bal_acc
            best_model_name = model_name
            best_phase_name = phase_name
            best_phase = phase_dict
            best_X_train = X_tr
            best_y_train = y_tr

print(f"Best Model: {best_model_name} with {best_phase_name}")
print(f"Test Balanced Accuracy: {best_score:.4f}")
print(f"Test Macro F1: {best_phase[best_model_name]['f1']:.4f}")

print("\n" + "=" * 70)
print(f"CROSS-VALIDATION — {best_model_name.upper()} + {best_phase_name.upper()}")
print("=" * 70)

# Recreate the best model for cross-validation
if best_phase_name == "CLASS WEIGHTING":
    best_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42,
                                       class_weight='balanced') if best_model_name == 'Random Forest' else None
    if best_model_name == 'XGBoost':
        best_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                  eval_metric='mlogloss', use_label_encoder=False)
    elif best_model_name == 'CatBoost':
        best_model = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_state=42)
    cv_with_weights = True
else:
    if best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    elif best_model_name == 'XGBoost':
        best_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                  eval_metric='mlogloss', use_label_encoder=False)
    elif best_model_name == 'CatBoost':
        best_model = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_state=42)
    cv_with_weights = False

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if cv_with_weights and best_phase_name == "CLASS WEIGHTING":
    bal_scores = cross_val_score(best_model, best_X_train, best_y_train, cv=cv, scoring='balanced_accuracy')
    f1_scores  = cross_val_score(best_model, best_X_train, best_y_train, cv=cv,
                                 scoring=make_scorer(f1_score, average='macro'))
else:
    bal_scores = cross_val_score(best_model, best_X_train, best_y_train, cv=cv, scoring='balanced_accuracy')
    f1_scores  = cross_val_score(best_model, best_X_train, best_y_train, cv=cv,
                                 scoring=make_scorer(f1_score, average='macro'))

print(f"Balanced Accuracy: {bal_scores.mean():.4f} ± {bal_scores.std():.4f}  |  scores: {bal_scores.round(4)}")
print(f"Macro F1:          {f1_scores.mean():.4f} ± {f1_scores.std():.4f}  |  scores: {f1_scores.round(4)}")

best_model.fit(best_X_train, best_y_train)
final_pred    = best_model.predict(X_test_arr)
final_bal_acc = balanced_accuracy_score(y_test_np, final_pred)
final_f1      = f1_score(y_test_np, final_pred, average='macro')
print(f"\nTest set: Balanced Acc = {final_bal_acc:.4f}, Macro F1 = {final_f1:.4f}")