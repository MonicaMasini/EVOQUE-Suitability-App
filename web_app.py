# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_curve, auc, mean_absolute_error, mean_squared_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='EVOQUE Suitability Prediction', layout='centered')

# Brand-inspired styling (Edwards red/gray palette)
EDWARDS_RED = '#C00000'
EDWARDS_ORANGE = '#ED7D31'
EDWARDS_GREEN = '#228B22'
EDWARDS_GRAY = '#6D6E71'

# Centered layout + padding
st.markdown(f'''
    <style>
    .main {{
        padding: 2rem 3rem;
    }}
    .big-title {{
        font-family: Arial, sans-serif; color:{EDWARDS_GRAY}; text-align:center; font-size: 2.0rem; font-weight: 700;
    }}
    .sub-title {{
        font-family: Arial, sans-serif; color:{EDWARDS_GRAY}; text-align:center; font-size: 1.2rem; margin-top: 0.5rem;
    }}
    .pred-box {{
        text-align: center; margin-top: 1.2rem; padding: 1.0rem; border-radius: 8px; border: 1px solid #ddd;
        font-family: Arial, sans-serif;
    }}
    .pred-label {{ font-weight: 800; font-size: 1.3rem; }}
    .pred-sub {{ font-style: italic; color: #333; font-size: 0.95rem; }}
    .tooltip {{ font-size: 0.85rem; color: #444; margin-top: 0.25rem; text-align:center; }}
    </style>
''', unsafe_allow_html=True)

st.markdown('<div class="big-title">EVOQUE Suitability Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Echo measurements</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_excel(path)
    cols = {c:str(c).strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    sl_col, ap_col, y_col = None, None, 'Decision'
    for col in list(df.columns):
      cl = col.lower()
      if cl.startswith('s') and 'dimension' in cl and 'a-p' not in cl and 'ap' not in cl:
        sl_col = col
      if ('a-p' in cl or 'ap ' in cl or ' a-p' in cl) and 'dimension' in cl:
        ap_col = col
      if cl == 'decision':
        y_col = col
    sl_col = sl_col or 'S-L dimension (mm)'
    ap_col = ap_col or 'A-P Dimension (mm)'
    work = df[[y_col, sl_col, ap_col]].copy()
    work.columns = ['Decision','SL','AP']
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work['ratio'] = work['SL'] / np.where(work['AP']==0, np.nan, work['AP'])
    sl_mean, sl_std = work['SL'].mean(), work['SL'].std(ddof=0)
    ap_mean, ap_std = work['AP'].mean(), work['AP'].std(ddof=0)
    work['z_SL'] = (work['SL'] - sl_mean) / (sl_std if sl_std>0 else 1.0)
    work['z_AP'] = (work['AP'] - ap_mean) / (ap_std if ap_std>0 else 1.0)
    work = work.dropna()
    work['y_suitable'] = (work['Decision'].str.strip().str.lower()=='suitable').astype(int)
    work['y_screenfail'] = 1 - work['y_suitable']
    return work, (sl_mean, sl_std, ap_mean, ap_std)

work, stats = load_data('Echo_Annular_Measurements.xlsx')
sl_mean, sl_std, ap_mean, ap_std = stats

X = work[['SL','AP','ratio','z_SL','z_AP']].values
y_suitable = work['y_suitable'].values

X_train, X_val, ytr_s, yval_s = train_test_split(
    X, y_suitable, test_size=0.3, random_state=42, stratify=(1 - y_suitable)
)

from collections import Counter
cnt = Counter(ytr_s)
w_suitable = 1.0 / cnt[1]
w_sfail = (1.0 / cnt[0]) * 2.0
class_weight = {1: w_suitable, 0: w_sfail}

rf = RandomForestClassifier(
    n_estimators=400,
    criterion='log_loss',
    max_depth=None,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
)

param_dist = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [None, 6, 8, 10],
    'min_samples_leaf': [1,2,3,5,8],
    'max_features': ['sqrt','log2']
}

def ap_sf_scorer(estimator, Xv, yv):
    ps = estimator.predict_proba(Xv)[:,1]
    psf = 1.0 - ps
    return average_precision_score(1 - yv, psf)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=12, scoring=ap_sf_scorer,
                        cv=cv, random_state=42, n_jobs=-1)
rs.fit(X_train, ytr_s)
rf_best = rs.best_estimator_

method = 'isotonic' if int(work['y_screenfail'].sum()) >= 30 else 'sigmoid'
cal_rf = CalibratedClassifierCV(rf_best, method=method, cv=5)
cal_rf.fit(X_train, ytr_s)

p_s_val = cal_rf.predict_proba(X_val)[:,1]

candidate_lows = np.linspace(0.15, 0.4, 6)
candidate_highs = np.linspace(0.6, 0.85, 6)
best = None
for tl in candidate_lows:
    for th in candidate_highs:
        if tl >= th:
            continue
        pred = np.where(p_s_val >= th, 'Suitable', np.where(p_s_val <= tl, 'Screen Fail', 'CT Recommended'))
        y_true_sf = (yval_s==0).astype(int)
        y_pred_sf = (pred=='Screen Fail').astype(int)
        tp = ((y_pred_sf==1) & (y_true_sf==1)).sum()
        fn = ((y_pred_sf==0) & (y_true_sf==1)).sum()
        recall_sf = tp / (tp+fn) if (tp+fn)>0 else 0
        y_true_s = (yval_s==1).astype(int)
        y_pred_s = (pred=='Suitable').astype(int)
        tp_s = ((y_pred_s==1) & (y_true_s==1)).sum()
        fp_s = ((y_pred_s==1) & (y_true_s==0)).sum()
        precision_s = tp_s / (tp_s+fp_s) if (tp_s+fp_s)>0 else 0
        frac_ct = (pred=='CT Recommended').mean()
        score = 2*recall_sf + precision_s - frac_ct
        if (best is None) or (score > best['score']):
            best = {'t_low': float(tl), 't_high': float(th), 'recall_sf': float(recall_sf), 'precision_s': float(precision_s), 'frac_ct': float(frac_ct), 'score': float(score)}

suitable = work[work['y_suitable']==1]
p95_sl = float(np.percentile(suitable['SL'], 95))
p95_ap = float(np.percentile(suitable['AP'], 95))
corridor_sl_min = max(55.0, p95_sl)
corridor_ap_max = min(49.0, p95_ap)

col1, col2 = st.columns(2)
with col1:
    sl_in = st.number_input('S-L Diameter (mm)', min_value=20.0, max_value=80.0, value=46.0, step=1.0, key='sl_input_v1')
with col2:
    ap_in = st.number_input('A-P Diameter (mm)', min_value=20.0, max_value=80.0, value=44.0, step=1.0, key='ap_input_v1')

ratio_in = sl_in / ap_in if ap_in>0 else np.nan
zSL_in = (sl_in - sl_mean) / (sl_std if sl_std>0 else 1.0)
zAP_in = (ap_in - ap_mean) / (ap_std if ap_std>0 else 1.0)
X_in = np.array([[sl_in, ap_in, ratio_in, zSL_in, zAP_in]])

if st.button('Predict', type='primary', use_container_width=True, key='predict_btn_v1'):
    p_s = float(cal_rf.predict_proba(X_in)[:,1][0])
    category = 'CT Recommended'
    color = EDWARDS_ORANGE
    if p_s >= best['t_high']:
        category = 'Suitable Annular Dimensions'
        color = EDWARDS_GREEN
    elif p_s <= best['t_low']:
        category = 'Unsuitable Annular Dimensions'
        color = EDWARDS_RED
    if category == 'Unsuitable Annular Dimensions' and (sl_in >= corridor_sl_min and ap_in <= corridor_ap_max):
        category = 'Indeterminate Annular Dimensions'
        color = EDWARDS_ORANGE
    st.markdown(f'<div class="pred-box"><div class="pred-label" style="color:{color}">{category}</div>'
                f'<div class="pred-sub">(confidence: {p_s*100:.1f}% p(Suitable))</div></div>', unsafe_allow_html=True)
    if category.startswith('Suitable'):
        st.markdown('<div class="tooltip">Suitable Annular Dimensions — S-L and A-P echo diameters fall within typical annular ranges for EVOQUE. Final therapy decision requires complete imaging and clinical assessment.</div>', unsafe_allow_html=True)
    elif category.startswith('Unsuitable'):
        st.markdown('<div class="tooltip">Unsuitable Annular Dimensions — S-L and A-P echo diameters are outside typical annular ranges for EVOQUE.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tooltip">Indeterminate Annular Dimensions — S-L and A-P echo diameters are near decision boundaries or conflicting; proceed with CT and full assessment for final suitability decision and sizing.</div>', unsafe_allow_html=True)

st.divider()
st.subheader('Model metrics & downloads')

if st.button('Build metrics & predictions workbook', key='build_metrics_btn_v1'):
    p_s_all = cal_rf.predict_proba(X)[:,1]
    pred = np.where(p_s_all >= best['t_high'], 'Suitable Annular Dimensions', np.where(p_s_all <= best['t_low'], 'Unsuitable Annular Dimensions', 'Indeterminate Annular Dimensions'))
    pred = np.where((pred=='Unsuitable Annular Dimensions') & (work['SL'].values>=corridor_sl_min) & (work['AP'].values<=corridor_ap_max), 'Indeterminate Annular Dimensions', pred)
    out_df = work.copy()
    out_df = out_df.assign(p_Suitable=p_s_all, Predicted_Category=pred)
    metrics = {
        'calibration_method': method,
        't_low': best['t_low'],
        't_high': best['t_high']
    }
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        out_df.to_excel(writer, index=False, sheet_name='Predictions')
        pd.DataFrame(list(metrics.items()), columns=['Metric','Value']).to_excel(writer, index=False, sheet_name='Metrics')
    st.download_button('Download metrics & predictions workbook', data=bio.getvalue(), file_name='EVOQUE_model_predictions_and_metrics.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_workbook_v1')

if st.button('Build & download plots', key='build_plots_btn_v1'):
    p_s_all = cal_rf.predict_proba(X)[:,1]
    pred = np.where(p_s_all >= best['t_high'], 'Suitable', np.where(p_s_all <= best['t_low'], 'Screen Fail', 'CT Recommended'))
    fig1, ax1 = plt.subplots(figsize=(8,5))
    for cls, color in [('Screen Fail',EDWARDS_RED), ('CT Recommended',EDWARDS_ORANGE), ('Suitable',EDWARDS_GREEN)]:
        mask = pred==cls
        if np.any(mask):
            sns.kdeplot(p_s_all[mask], ax=ax1, label=f"{cls}", fill=True, common_norm=False)
    ax1.axvline(best['t_low'], color=EDWARDS_ORANGE, linestyle='--', label='t_low')
    ax1.axvline(best['t_high'], color=EDWARDS_GREEN, linestyle='--', label='t_high')
    ax1.set_title('Probability of Suitable — Distribution by Prediction Category')
    ax1.set_xlabel('p(Suitable)')
    ax1.legend()
    b1 = BytesIO(); fig1.savefig(b1, format='png', dpi=160); st.download_button('Download probability distribution plot', data=b1.getvalue(), file_name='probability_distribution_by_category.png', mime='image/png', key='dl_plot1_v1')

    fig2, ax2 = plt.subplots(figsize=(6,6))
    sc = ax2.scatter(work['SL'], work['AP'], c=p_s_all, cmap='RdYlGn', s=30, edgecolor='k', alpha=0.8)
    ax2.set_xlabel('S-L (mm)'); ax2.set_ylabel('A-P (mm)'); ax2.set_title('SL vs AP — Colored by p(Suitable)')
    cb = plt.colorbar(sc, ax=ax2); cb.set_label('p(Suitable)')
    ax2.axvline({corridor_sl_min}, color='k', linestyle=':')
    ax2.axhline({corridor_ap_max}, color='k', linestyle=':')
    b2 = BytesIO(); fig2.savefig(b2, format='png', dpi=160); st.download_button('Download SL-AP scatter plot', data=b2.getvalue(), file_name='scatter_sl_ap_probability.png', mime='image/png', key='dl_plot2_v1')

    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    p_sf_all = 1.0 - p_s_all
    fpr, tpr, _ = roc_curve(1 - y_suitable, p_sf_all)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(1 - y_suitable, p_sf_all)
    pr_auc = auc(rec, prec)
    fig3, ax3 = plt.subplots(1,2, figsize=(12,5))
    ax3[0].plot(fpr, tpr, color=EDWARDS_RED); ax3[0].plot([0,1],[0,1],'k--', alpha=0.5)
    ax3[0].set_title(f'ROC (Screen Fail) — AUC={roc_auc:.3f}'); ax3[0].set_xlabel('FPR'); ax3[0].set_ylabel('TPR')
    ax3[1].plot(rec, prec, color=EDWARDS_RED); ax3[1].set_title(f'PR (Screen Fail) — AUC={pr_auc:.3f}'); ax3[1].set_xlabel('Recall'); ax3[1].set_ylabel('Precision')
    b3 = BytesIO(); fig3.savefig(b3, format='png', dpi=160); st.download_button('Download performance curves', data=b3.getvalue(), file_name='performance_curves.png', mime='image/png', key='dl_plot3_v1')

    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_suitable, p_s_all, n_bins=10, strategy='uniform')
    fig4, ax4 = plt.subplots(figsize=(5,5))
    ax4.plot(prob_pred, prob_true, 'o-', label='Model'); ax4.plot([0,1],[0,1],'k--', label='Perfect')
    ax4.set_xlabel('Mean predicted p(Suitable)'); ax4.set_ylabel('Observed frequency'); ax4.set_title('Calibration Curve')
    ax4.legend()
    b4 = BytesIO(); fig4.savefig(b4, format='png', dpi=160); st.download_button('Download calibration curve', data=b4.getvalue(), file_name='calibration_curve.png', mime='image/png', key='dl_plot4_v1')

if st.button('Generate SHAP (on demand)', key='shap_btn_v1'):
    try:
        import shap
        explainer = shap.TreeExplainer(cal_rf)
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(7,4))
        shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X, feature_names=['SL','AP','ratio','z_SL','z_AP'], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning('SHAP is not available in this environment. Please install shap to use this feature.')

st.caption(f"Using thresholds: t_low={best['t_low']:.2f} (Screen fail), t_high={best['t_high']:.2f} (Suitable). Calibration: {method}.")
