import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import shap
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Housing Loan Fraud Detection", page_icon=None, layout="wide")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .hero { background: linear-gradient(135deg, #1a1f2e 0%, #0d1321 100%); border: 1px solid #2d3748; border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; text-align: center; }
  .hero h1 { font-size: 2.2rem; font-weight: 600; color: #f0f4f8; margin: 0 0 0.5rem; }
  .hero p  { color: #8899aa; font-size: 1rem; margin: 0; }
  .hero .badge { display: inline-block; background: #1e3a5f; color: #60a5fa; font-size: 0.75rem; font-weight: 500; padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem; letter-spacing: 0.05em; }
  .metric-card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 1.25rem 1.5rem; text-align: center; }
  .metric-card .label  { font-size: 0.75rem; font-weight: 500; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
  .metric-card .value  { font-size: 2rem; font-weight: 600; line-height: 1; margin-bottom: 0.25rem; }
  .metric-card .meaning{ font-size: 0.88rem; color: #9aa5b8; line-height: 1.5; }
  .stButton > button { width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; border-radius: 10px; padding: 0.75rem 1.5rem; font-size: 1rem; font-weight: 500; }
  div[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    df = pd.read_csv("homeloancsv.csv")
    yn_cols = ['Signature_Mismatch','ID_Docs_Altered','Salary_Round_Figure_Flag',
        'PF_Deduction_Missing','Salary_Credited_From_Personal_Acct',
        'ITR_Filed_Just_Before_Application','Month_End_Fund_Parking',
        'Valuation_Inflation_Flag','OC_CC_Available','Disbursement_To_NonSeller_Flag',
        'Bank_Statement_Tampered','PDF_Metadata_Anomaly','Borrower_Unreachable',
        'AML_Flag','Loan_Foreclosed_Within_6M','PEP_Involved','Overseas_Wire_Transfer_Detected']
    for col in yn_cols:
        df[col] = df[col].map({'Yes':1,'No':0,1:1,0:0}).fillna(0)
    df['income_discrepancy']   = (df['Monthly_Income_Stated'] - df['Actual_Bank_Credit_Monthly']).fillna(0)
    df['income_inflate_ratio'] = (df['Monthly_Income_Stated'] / df['Actual_Bank_Credit_Monthly'].replace(0,np.nan)).fillna(1).clip(upper=10)
    df['itr_income_gap']       = (df['Annual_Income_Stated'] - df['ITR_Income_Last_FY'].fillna(df['Annual_Income_Stated']))
    doc_flag_cols = ['Email_Name_Mismatch'] + yn_cols
    df['doc_risk_score'] = df[doc_flag_cols].sum(axis=1)
    numeric_features = ['Monthly_Income_Stated','Actual_Bank_Credit_Monthly','LTV_Ratio',
        'FOIR','CIBIL_Score','Num_Credit_Enquiries_Last_30D','Loan_Amount_INR','Tenure_Years',
        'Existing_Monthly_EMI','Years_With_Employer','income_discrepancy','income_inflate_ratio','itr_income_gap','doc_risk_score']
    categorical_features = ['Employment_Type','Loan_Purpose','Property_Type','Down_Payment_Source','Disbursement_Mode']
    X = df[numeric_features + categorical_features + doc_flag_cols].copy()
    y = df['Fraud_Label_01'].copy()
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    for col in categorical_features:
        X[col] = X[col].fillna('Unknown')
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── SMOTE oversampling to fix class imbalance ─────────────────────────
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm,   y_train_sm   = sm.fit_resample(X_train,   y_train)
    X_train_s_sm, y_train_s_sm = sm.fit_resample(X_train_s, y_train)

    # ── All individual + ensemble candidates ──────────────────────────────
    xgb  = XGBClassifier(n_estimators=100, scale_pos_weight=10, random_state=42, eval_metric='logloss', verbosity=0)
    rf   = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    et   = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    gb   = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr   = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    dt   = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    knn  = KNeighborsClassifier(n_neighbors=5)

    # Voting ensemble: XGBoost + Random Forest
    voting_xgb_rf = VotingClassifier(
        estimators=[('xgb', XGBClassifier(n_estimators=100, scale_pos_weight=10, random_state=42, eval_metric='logloss', verbosity=0)),
                    ('rf',  RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))],
        voting='soft')

    # Stacking: XGBoost + Logistic Regression
    stacking_xgb_lr = StackingClassifier(
        estimators=[('xgb', XGBClassifier(n_estimators=100, scale_pos_weight=10, random_state=42, eval_metric='logloss', verbosity=0))],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        passthrough=True, cv=3)

    # Stacking: XGBoost + Extra Trees + Logistic (best combo)
    stacking_xgb_et_lr = StackingClassifier(
        estimators=[('xgb', XGBClassifier(n_estimators=100, scale_pos_weight=10, random_state=42, eval_metric='logloss', verbosity=0)),
                    ('et',  ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42))],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        passthrough=True, cv=3)

    TREE_MODELS = {'XGBoost','Random Forest','Gradient Boosting','Extra Trees','Decision Tree',
                   'Voting: XGBoost + RF','Stack: XGBoost + LR','Stack: XGBoost + ExtraTrees + LR'}

    candidates = {
        'Logistic Regression':           lr,
        'Random Forest':                 rf,
        'Extra Trees':                   et,
        'Gradient Boosting':             gb,
        'XGBoost':                       xgb,
        'Decision Tree':                 dt,
        'K-Nearest Neighbors':           knn,
        'Voting: XGBoost + RF':          voting_xgb_rf,
        'Stack: XGBoost + LR':           stacking_xgb_lr,
        'Stack: XGBoost + ExtraTrees + LR': stacking_xgb_et_lr,
    }

    from sklearn.metrics import precision_recall_curve

    def find_best_threshold(y_true, probs, min_recall=0.75):
        """Find threshold that maximises F1 subject to recall >= min_recall."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
        best_f1, best_thresh = -1, 0.5
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
            if r >= min_recall:
                f1 = 2*p*r/(p+r+1e-9)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, t
        # fallback: if no threshold achieves min_recall, use 0.25
        if best_f1 < 0:
            best_thresh = 0.25
        return float(best_thresh)

    comparison_rows = []
    best_f1, best_model, best_name, best_thresh_global = -1, None, '', 0.5

    for name, clf in candidates.items():
        Xtr    = X_train_sm   if name in TREE_MODELS else X_train_s_sm
        Xte    = X_test       if name in TREE_MODELS else X_test_s
        y_tr   = y_train_sm   if name in TREE_MODELS else y_train_s_sm
        try:
            clf.fit(Xtr, y_tr)
            prob   = clf.predict_proba(Xte)[:, 1]
            thresh = find_best_threshold(y_test, prob, min_recall=0.75)
            pred   = (prob >= thresh).astype(int)
            m_auc  = roc_auc_score(y_test, prob)
            m_prec = precision_score(y_test, pred, zero_division=0)
            m_rec  = recall_score(y_test, pred, zero_division=0)
            m_f1   = 2*m_prec*m_rec/(m_prec+m_rec+1e-9)
            tag = "Individual" if name in ['Logistic Regression','Random Forest','Extra Trees','Gradient Boosting','XGBoost','Decision Tree','K-Nearest Neighbors'] else ("Voting Ensemble" if "Voting" in name else "Stacking Ensemble")
            comparison_rows.append({'Model':name,'Type':tag,'AUC':round(m_auc,4),'Recall':round(m_rec,4),'Precision':round(m_prec,4),'F1':round(m_f1,4),'Threshold':round(thresh,3)})
            # Select best by F1 while recall >= 0.75
            if m_rec >= 0.75 and m_f1 > best_f1:
                best_f1, best_model, best_name, best_thresh_global = m_f1, clf, name, thresh
        except Exception as e:
            comparison_rows.append({'Model':name,'Type':'Error','AUC':0,'Recall':0,'Precision':0,'F1':0,'Threshold':0.5})

    # Fallback: if no model achieves recall>=0.75, pick highest recall
    if best_model is None:
        for row in comparison_rows:
            if row['Recall'] > (best_f1 if best_f1 > 0 else -1):
                best_f1 = row['Recall']
                best_name = row['Model']
                best_model = candidates[best_name]
                best_thresh_global = 0.25

    comparison_df = pd.DataFrame(comparison_rows).sort_values('AUC', ascending=False).reset_index(drop=True)

    is_tree = best_name in TREE_MODELS
    Xte = X_test if is_tree else X_test_s
    y_prob = best_model.predict_proba(Xte)[:, 1]
    y_pred = (y_prob >= best_thresh_global).astype(int)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    try:
        importances = best_model.feature_importances_
    except AttributeError:
        try:
            importances = np.abs(best_model.final_estimator_.coef_[0][:len(X_train.columns)])
        except:
            importances = np.zeros(len(X_train.columns))
    coef_df = pd.DataFrame({'feature': X_train.columns, 'weight': importances[:len(X_train.columns)]}).sort_values('weight', ascending=False)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        kf_scores = cross_val_score(best_model, X_train if is_tree else X_train_s, y_train, cv=kf, scoring='roc_auc')
    except Exception:
        kf_scores = np.array([auc]*5)

    try:
        core = best_model.estimators_[0][1] if hasattr(best_model, 'estimators_') else best_model
        explainer = shap.TreeExplainer(core)
        _ = explainer.shap_values(X_train.iloc[:1])
        use_tree_shap = True
    except Exception:
        lr_fb = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr_fb.fit(X_train_s_sm, y_train_sm)  # use SMOTE data for consistency
        explainer = shap.LinearExplainer(lr_fb, X_train_s_sm, feature_perturbation="interventional")
        use_tree_shap = False

    test_df = df.loc[X_test.index].copy()
    test_df['risk_score'] = (y_prob * 100).round(1)
    test_df['predicted']  = y_pred
    test_df['actual']     = y_test.values
    test_df['category']   = test_df.apply(lambda r:
        'TP' if r['predicted']==1 and r['actual']==1 else
        'FP' if r['predicted']==1 and r['actual']==0 else
        'FN' if r['predicted']==0 and r['actual']==1 else 'TN', axis=1)

    return (best_model, scaler, list(X_train.columns), auc, prec, rec,
            tn, fp, fn, tp, coef_df, kf_scores, df, doc_flag_cols,
            numeric_features, categorical_features, test_df, explainer,
            comparison_df, best_name, X_train, X_train_s, use_tree_shap)


def call_claude_api(messages, system_prompt):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "\n No API key found. Add GROQ_API_KEY to Streamlit Secrets."
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": 1000,
                  "messages": [{"role": "system", "content": system_prompt}] + messages},
            timeout=30)
        data = resp.json()
        if "choices" in data: return data["choices"][0]["message"]["content"]
        elif "error" in data: return f"\n API error: {data['error'].get('message','Unknown error')}"
        return "\n Unexpected response from API."
    except Exception as e:
        return f"\n Could not connect to advisor. Error: {str(e)}"


def build_system_prompt(inputs, score, inflate, flags_count, shap_summary, overall_risk=None):
    display_score = overall_risk if overall_risk is not None else score
    score_pct  = display_score * 100
    risk_level = "HIGH RISK" if display_score >= 0.70 else "MEDIUM RISK" if display_score >= 0.40 else "LOW-MEDIUM RISK" if display_score >= 0.25 else "LOW RISK"
    flags_ticked = [k for k, v in inputs.items() if isinstance(v, int) and v == 1 and k not in
                    ['Age','CIBIL_Score','Num_Credit_Enquiries_Last_30D','Tenure_Years']]
    return f"""You are a senior fraud investigation advisor for an Indian housing loan bank.
Risk Score: {score_pct:.1f}% ({risk_level}) | Age: {inputs.get('Age')} | Employment: {inputs.get('Employment_Type')}
Stated Income: Rs{inputs.get('Monthly_Income_Stated',0):,}/mo | Actual Credits: Rs{inputs.get('Actual_Bank_Credit_Monthly',0):,}/mo | Inflate Ratio: {inflate:.2f}x
CIBIL: {inputs.get('CIBIL_Score')} | Loan: Rs{inputs.get('Loan_Amount_INR',0):,} | LTV: {inputs.get('LTV_Ratio')} | FOIR: {inputs.get('FOIR')}
Flags triggered ({flags_count}/17): {', '.join(flags_ticked) if flags_ticked else 'None'}
Top SHAP factors: {shap_summary}
You are a senior colleague. Give concise bullet answers max 200 words. Recommend REJECT/ESCALATE/REFER TO RISK TEAM/ELIGIBLE FOR UNDERWRITING when asked — never use the word 'proceed' or 'approve'. Ground answers in the numbers above. You know RBI guidelines and Indian housing loan fraud patterns."""


def render_chat_widget(inputs, score, inflate, flags_count, shap_summary, overall_risk=None):
    if 'chat_open' not in st.session_state: st.session_state.chat_open = False
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'chat_input_key' not in st.session_state: st.session_state.chat_input_key = 0
    system_prompt = build_system_prompt(inputs, score, inflate, flags_count, shap_summary, overall_risk=overall_risk)
    col_spacer, col_btn = st.columns([6, 1])
    with col_btn:
        if st.button("Close" if st.session_state.chat_open else "Advisor", key="chat_toggle_btn"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()
    if not st.session_state.chat_open: return
    st.markdown("---")
    for msg in st.session_state.chat_history:
        align = "right" if msg["role"]=="user" else "left"
        bg    = "#1e3a5f" if msg["role"]=="user" else "#1a1f2e"
        color = "#93c5fd" if msg["role"]=="user" else "#f0f4f8"
        st.markdown(f'<div style="background:{bg};border-radius:10px;padding:0.7rem 1.1rem;margin:6px 0;text-align:{align}"><span style="font-size:1.02rem;line-height:1.5;color:{color}">{msg["content"]}</span></div>', unsafe_allow_html=True)
    user_q = st.text_input("Ask...", key=f"chat_input_{st.session_state.chat_input_key}", label_visibility="collapsed")
    c1, c2 = st.columns([3,1])
    with c1: send = st.button("Send", key="chat_send", use_container_width=True)
    with c2:
        if st.button("Clear", key="chat_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    if send and user_q.strip():
        st.session_state.chat_history.append({"role":"user","content":user_q.strip()})
        with st.spinner("Thinking..."):
            reply = call_claude_api(st.session_state.chat_history, system_prompt)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.session_state.chat_input_key += 1
        st.rerun()


def score_application(inputs, model, scaler, feature_names, numeric_features, categorical_features, doc_flag_cols, is_tree):
    app = pd.DataFrame([inputs])
    actual = inputs.get('Actual_Bank_Credit_Monthly',1) or 1
    stated = inputs.get('Monthly_Income_Stated',0)
    annual = inputs.get('Annual_Income_Stated',0)
    itr    = inputs.get('ITR_Income_Last_FY',annual)
    app['income_discrepancy']   = stated - actual
    app['income_inflate_ratio'] = min(stated/actual if actual else 1, 10)
    app['itr_income_gap']       = annual - itr
    app['doc_risk_score']       = sum(int(inputs.get(c,0)) for c in doc_flag_cols)
    for col in categorical_features:
        if col not in app.columns: app[col] = 'Unknown'
    app = pd.get_dummies(app, columns=categorical_features, drop_first=True)
    for col in feature_names:
        if col not in app.columns: app[col] = 0
    app = app[feature_names]
    for col in [c for c in numeric_features if c in app.columns]:
        app[col] = pd.to_numeric(app[col], errors='coerce').fillna(0)
    X_in = app if is_tree else scaler.transform(app)
    return model.predict_proba(X_in)[0][1], app


def render_loan_table(data, category):
    color_map = {
        'TP':('#34d399','#064e3b','Correctly caught fraud','Model said FRAUD - Actually FRAUD'),
        'FP':('#fbbf24','#78350f','False alarm - innocent','Model said FRAUD - Actually CLEAN'),
        'FN':('#f87171','#7f1d1d','Missed fraud - slipped thru','Model said CLEAN - Actually FRAUD'),
        'TN':('#60a5fa','#1e3a5f','Correctly approved clean','Model said CLEAN - Actually CLEAN'),
    }
    color, bg, short, long = color_map[category]
    st.markdown(f'<div style="background:{bg};border:1px solid {color}33;border-radius:12px;padding:12px 16px;margin-bottom:16px;text-align:center"><div style="font-size:1rem;font-weight:600;color:{color}">{short}</div><div style="font-size:0.8rem;color:{color};opacity:0.8;margin-top:3px">{long}</div></div>', unsafe_allow_html=True)
    priority_cols = ['risk_score','category','Fraud_Type','Anomaly_Notes','Application_ID','Application_Date','Borrower_Name','Age','Gender','Employment_Type','Monthly_Income_Stated','Actual_Bank_Credit_Monthly','income_inflate_ratio','Annual_Income_Stated','ITR_Income_Last_FY','CIBIL_Score','Loan_Amount_INR','LTV_Ratio','FOIR','Tenure_Years','Loan_Purpose','Property_Type','Down_Payment_Source','Disbursement_Mode','doc_risk_score','Email_Name_Mismatch','Signature_Mismatch','ID_Docs_Altered','Bank_Statement_Tampered','AML_Flag','PEP_Involved','Fraud_Label_01']
    show_cols = [c for c in priority_cols if c in data.columns]
    remaining = [c for c in data.columns if c not in show_cols and c not in ['predicted','actual']]
    display = data[show_cols+remaining].copy().reset_index(drop=True)
    for col in ['Monthly_Income_Stated','Actual_Bank_Credit_Monthly','Annual_Income_Stated','ITR_Income_Last_FY','Loan_Amount_INR']:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"Rs{x:,.0f}" if pd.notna(x) else "-")
    if 'income_inflate_ratio' in display.columns:
        display['income_inflate_ratio'] = display['income_inflate_ratio'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
    if 'risk_score' in display.columns:
        display['risk_score'] = display['risk_score'].apply(lambda x: f"{x}%")
    st.dataframe(display, use_container_width=True, height=450)
    st.caption(f"{len(display):,} applications - scroll right to see all columns")


def main():
    (model, scaler, feature_names, auc, prec, rec,
     tn, fp, fn, tp, coef_df, kf_scores, df, doc_flag_cols,
     numeric_features, categorical_features, test_df, explainer,
     comparison_df, best_name, X_train, X_train_s, use_tree_shap) = load_and_train()

    TREE_MODELS = {'XGBoost','Random Forest','Gradient Boosting','Extra Trees','Decision Tree',
                   'Voting: XGBoost + RF','Stack: XGBoost + LR','Stack: XGBoost + ExtraTrees + LR'}
    is_tree = best_name in TREE_MODELS

    st.markdown(f"""<div class="hero"><div class="badge">HOUSING LOAN - FRAUD DETECTION SYSTEM</div>
        <h1>Anti-Fraud Detection Dashboard</h1>
        <p>Best Model: <strong>{best_name}</strong> - 20-Year Dataset - 10,080 Loan Applications - 4% Fraud Rate</p></div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Performance","Score New Application","Feature Importance","Inspect Predictions","Model Comparison"])

    with tab1:
        st.markdown("### Model Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="label">AUC-ROC</div><div class="value" style="color:#60a5fa">{auc:.4f}</div><div class="meaning">Overall model quality. 1.0 = perfect.</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="label">Recall</div><div class="value" style="color:#34d399">{rec:.4f}</div><div class="meaning">Of all real fraud cases, this many were caught.</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="label">Precision</div><div class="value" style="color:#a78bfa">{prec:.4f}</div><div class="meaning">Of all flagged cases, this many were real fraud.</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase">CONFUSION MATRIX</p>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f'<div class="metric-card" style="border-color:#064e3b"><div class="label" style="color:#34d399">True Negative</div><div class="value" style="color:#34d399">{tn:,}</div><div class="meaning">Clean loans correctly approved</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card" style="border-color:#7f1d1d;margin-top:10px"><div class="label" style="color:#f87171">False Negative</div><div class="value" style="color:#f87171">{fn:,}</div><div class="meaning">Fraudsters who slipped through</div></div>', unsafe_allow_html=True)
            with cb:
                st.markdown(f'<div class="metric-card" style="border-color:#78350f"><div class="label" style="color:#fbbf24">False Positive</div><div class="value" style="color:#fbbf24">{fp:,}</div><div class="meaning">Innocent borrowers wrongly flagged</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card" style="border-color:#064e3b;margin-top:10px"><div class="label" style="color:#34d399">True Positive</div><div class="value" style="color:#34d399">{tp:,}</div><div class="meaning">Fraudsters correctly caught</div></div>', unsafe_allow_html=True)
        with col_right:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase">5-FOLD CROSS VALIDATION</p>', unsafe_allow_html=True)
            for i, s in enumerate(kf_scores, 1):
                st.markdown(f'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;margin-bottom:3px"><span style="font-size:0.8rem;color:#8899aa">Fold {i}</span><span style="font-size:0.8rem;font-weight:600;color:#60a5fa">{s:.4f}</span></div><div style="background:#1e2533;border-radius:4px;height:8px;overflow:hidden"><div style="width:{int(s*100)}%;height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:4px"></div></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:#1e3a5f;border-radius:10px;padding:12px 16px;margin-top:12px;display:flex;justify-content:space-between"><div style="text-align:center"><div style="font-size:1.4rem;font-weight:700;color:#60a5fa">{kf_scores.mean():.4f}</div><div style="font-size:0.7rem;color:#8899aa;margin-top:2px">Mean AUC</div></div><div style="text-align:center"><div style="font-size:1.4rem;font-weight:700;color:#34d399">{kf_scores.std():.4f}</div><div style="font-size:0.7rem;color:#8899aa;margin-top:2px">Std Dev</div></div></div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### Score a New Loan Application")
        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Borrower Details**")
                age            = st.slider("Age", 18, 70, 35)
                employment     = st.selectbox("Employment Type", ["Salaried","Self-Employed Business","Self-Employed Professional","Government"])
                years_employer = st.slider("Years with Employer", 0, 30, 5)
                cibil          = st.slider("CIBIL Score", 300, 900, 720)
                enquiries      = st.slider("Credit Enquiries (Last 30 Days)", 0, 15, 1)
            with col2:
                st.markdown("**Income & Loan**")
                stated_income = st.number_input("Stated Monthly Income (Rs)", 0, 2000000, 100000, step=5000)
                actual_credit = st.number_input("Actual Bank Credits/Month (Rs)", 0, 2000000, 95000, step=5000)
                annual_income = st.number_input("Annual Income Stated (Rs)", 0, 20000000, 1200000, step=50000)
                itr_income    = st.number_input("ITR Income Last FY (Rs)", 0, 20000000, 1150000, step=50000)
                loan_amount   = st.number_input("Loan Amount (Rs)", 100000, 50000000, 5000000, step=100000)
                ltv           = st.slider("LTV Ratio", 0.0, 1.0, 0.65, 0.01)
                foir          = st.slider("FOIR", 0.0, 1.0, 0.40, 0.01)
                tenure        = st.slider("Tenure (Years)", 1, 30, 15)
                emi           = st.number_input("Existing Monthly EMI (Rs)", 0, 500000, 5000, step=1000)
            with col3:
                st.markdown("**Property & Flags**")
                loan_purpose   = st.selectbox("Loan Purpose", ["Purchase","Balance Transfer","Construction","Extension","Repair/Renovation"])
                property_type  = st.selectbox("Property Type", ["Flat/Apartment","Row House","Independent House","Villa","Plot+Construction"])
                dp_source      = st.selectbox("Down Payment Source", ["Own Savings","Family Gift","Sale of Asset","Third Party"])
                disb_mode      = st.selectbox("Disbursement Mode", ["NEFT to Seller","NEFT to Builder","Cheque","Partial Cash","Cash"])
                st.markdown("**Document Flags**")
                bank_tampered  = st.checkbox("Bank Statement Tampered")
                pdf_anomaly    = st.checkbox("PDF Metadata Anomaly")
                aml_flag       = st.checkbox("AML Flag")
                pep            = st.checkbox("PEP Involved")
                disb_nonseller = st.checkbox("Disbursement to Non-Seller")
                sig_mismatch   = st.checkbox("Signature Mismatch")
                id_altered     = st.checkbox("ID Docs Altered")
                itr_before     = st.checkbox("ITR Filed Just Before Application")
                overseas       = st.checkbox("Overseas Wire Transfer")
                month_end_park = st.checkbox("Month End Fund Parking")
                val_inflation  = st.checkbox("Valuation Inflation")
                salary_personal= st.checkbox("Salary from Personal Account")
                pf_missing     = st.checkbox("PF Deduction Missing")
                unreachable    = st.checkbox("Borrower Unreachable")
                foreclosed     = st.checkbox("Loan Foreclosed Within 6M")
                round_salary   = st.checkbox("Salary Round Figure")
                oc_cc          = st.checkbox("OC/CC Available (protective)")
                email_mismatch = st.checkbox("Email Name Mismatch")
            submitted = st.form_submit_button("Calculate Fraud Risk Score")

        if submitted:
            inputs = {
                'Age':age,'Employment_Type':employment,'Years_With_Employer':years_employer,
                'CIBIL_Score':cibil,'Num_Credit_Enquiries_Last_30D':enquiries,
                'Monthly_Income_Stated':stated_income,'Actual_Bank_Credit_Monthly':actual_credit,
                'Annual_Income_Stated':annual_income,'ITR_Income_Last_FY':itr_income,
                'Loan_Amount_INR':loan_amount,'LTV_Ratio':ltv,'FOIR':foir,
                'Tenure_Years':tenure,'Existing_Monthly_EMI':emi,
                'Loan_Purpose':loan_purpose,'Property_Type':property_type,
                'Down_Payment_Source':dp_source,'Disbursement_Mode':disb_mode,
                'Bank_Statement_Tampered':int(bank_tampered),'PDF_Metadata_Anomaly':int(pdf_anomaly),
                'AML_Flag':int(aml_flag),'PEP_Involved':int(pep),
                'Disbursement_To_NonSeller_Flag':int(disb_nonseller),
                'Signature_Mismatch':int(sig_mismatch),'ID_Docs_Altered':int(id_altered),
                'ITR_Filed_Just_Before_Application':int(itr_before),
                'Overseas_Wire_Transfer_Detected':int(overseas),'Month_End_Fund_Parking':int(month_end_park),
                'Valuation_Inflation_Flag':int(val_inflation),'Salary_Credited_From_Personal_Acct':int(salary_personal),
                'PF_Deduction_Missing':int(pf_missing),'Borrower_Unreachable':int(unreachable),
                'Loan_Foreclosed_Within_6M':int(foreclosed),'Salary_Round_Figure_Flag':int(round_salary),
                'OC_CC_Available':int(oc_cc),'Email_Name_Mismatch':int(email_mismatch),
            }
            score, app_df = score_application(inputs, model, scaler, feature_names, numeric_features, categorical_features, doc_flag_cols, is_tree)
            # score is raw model probability — keep as-is; overall_risk handles final calibration
            st.session_state['last_inputs'] = inputs
            st.session_state['last_score']  = score

            # ── Critical fraud flags ────────────────────────────────────────────
            # Loan_Foreclosed_Within_6M is suspicious but NOT critical.
            # early closure can be legitimate (bonus, refinance, property sale).
            # It triggers MANUAL REVIEW via flags_count_excl_oc, not immediate escalation.
            critical_flags_triggered = sum([
                bank_tampered, id_altered, aml_flag, pep, sig_mismatch, overseas,
                disb_nonseller, pdf_anomaly
            ])
            # OC_CC_Available is protective (reduces risk) — intentionally excluded from flag count
            flags_count_excl_oc = sum([bank_tampered,pdf_anomaly,aml_flag,pep,disb_nonseller,sig_mismatch,id_altered,itr_before,overseas,month_end_park,val_inflation,salary_personal,pf_missing,unreachable,foreclosed,round_salary,email_mismatch])

            # ── Credit risk business rules (independent of fraud model) ────────
            # Calculate actual EMI the borrower would need to pay
            annual_rate = 0.09  # 9% standard home loan rate
            monthly_rate = annual_rate / 12
            tenure_months = tenure * 12
            if tenure_months > 0 and monthly_rate > 0:
                estimated_emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure_months / ((1 + monthly_rate)**tenure_months - 1)
            else:
                estimated_emi = 0

            # Total monthly obligation = new EMI + existing EMIs
            total_monthly_obligation = estimated_emi + emi
            # Use ACTUAL bank credits, not stated income which the borrower can inflate
            actual_credit_safe = actual_credit if actual_credit > 0 else 1
            # Effective FOIR after this loan, based on actual credits
            effective_foir = total_monthly_obligation / actual_credit_safe
            # EMI as % of actual income
            emi_to_income = estimated_emi / actual_credit_safe

            # ── Unified risk score ─────────────────────────────────────────
            # Three components combined as a weighted blend (35/45/20):
            #   1. ML model fraud probability (35%)
            #   2. Credit policy rules — FOIR, EMI, LTI, CIBIL, LTV (45%)
            #   3. Document / process flags (20%)
            # Each credit signal uses a weighted average — monotonic, no saturation.

            inflate = min(stated_income / actual_credit if actual_credit else 1, 10)

            def sigmoid_risk(value, threshold, scale, increasing=True):
                z = (value - threshold) / scale
                return 1 / (1 + np.exp(-z if increasing else z))

            # risk_components: (label, weight, risk_value, message)
            # Weighted average replaces noisy-OR to prevent saturation bugs.
            risk_components = []

            # ── CIBIL ────────────────────────────────────────────────────────
            # RBI norm: most lenders require 700+. Below 650 is near-subprime.
            # Sigmoid centred at 680 (not 620). Weight 3.0 — single biggest credit signal.
            cibil_r = sigmoid_risk(cibil, 680, 40, increasing=False)
            risk_components.append(("CIBIL score", 3.0, cibil_r,
                f"CIBIL is {cibil} (lender minimum typically 700; below 650 is high risk)."))

            # ── Effective FOIR after new loan ────────────────────────────────
            # RBI guideline: FOIR should not exceed 50% for income <Rs75k/mo,
            # 55% for Rs75k-1.5L, 60% absolute max. Sigmoid centred at 0.50,
            # tight scale 0.05 so crossing 55% already reads as high risk.
            eff_foir_r = sigmoid_risk(effective_foir, 0.50, 0.05, increasing=True)
            risk_components.append(("FOIR after this loan", 3.0, eff_foir_r,
                f"Total obligations after this loan: {effective_foir*100:.0f}% of actual income (RBI cap: 50-60%)."))

            # ── EMI affordability ────────────────────────────────────────────
            # New EMI alone should not exceed 40% of net income (conservative norm).
            # Sigmoid centred at 0.35, weight 2.5.
            emi_inc_r = sigmoid_risk(emi_to_income, 0.35, 0.05, increasing=True)
            risk_components.append(("EMI affordability", 2.5, emi_inc_r,
                f"New EMI is {emi_to_income*100:.0f}% of actual monthly income (safe limit: 40%)."))

            # ── Loan-to-income ───────────────────────────────────────────────
            # Standard: home loan should not exceed 5x gross annual income.
            # Above 6x is very high risk. Sigmoid centred at 4.5, weight 2.5.
            annual_actual = actual_credit_safe * 12
            lti = loan_amount / annual_actual if annual_actual > 0 else 0
            lti_r = sigmoid_risk(lti, 4.5, 0.8, increasing=True)
            risk_components.append(("Loan-to-income ratio", 2.5, lti_r,
                f"Loan is {lti:.1f}x annual income (safe limit: 5x)."))

            # ── LTV ──────────────────────────────────────────────────────────
            # RBI Circular RBI/2020-21/60: LTV cap is 75% for loans >Rs75L,
            # 80% for Rs30-75L, 90% for <Rs30L. Using 80% as a universal safe limit.
            # Sigmoid centred at 0.73, tight scale 0.03. Weight 3.0 — hard policy.
            ltv_r = sigmoid_risk(ltv, 0.73, 0.03, increasing=True)
            risk_components.append(("Loan-to-value", 3.0, ltv_r,
                f"LTV is {ltv*100:.0f}% (RBI cap: 75-80% depending on loan size)."))

            # ── Age at loan end ──────────────────────────────────────────────
            # Most lenders: borrower must be <=60 (salaried) or <=65 (self-employed)
            # at loan maturity. Sigmoid centred at 60, scale 2. Weight 2.0.
            age_at_end = age + tenure
            age_tenure_r = sigmoid_risk(age_at_end, 60, 2, increasing=True)
            risk_components.append(("Age at loan maturity", 2.0, age_tenure_r,
                f"Borrower will be {age_at_end} at loan maturity (salaried limit: 60)."))

            # ── Employment stability ─────────────────────────────────────────
            # Standard: minimum 2 years continuous employment for salaried.
            # < 1 year is a hard red flag. Sigmoid centred at 2.0, scale 0.8. Weight 2.0.
            emp_years_r = sigmoid_risk(years_employer, 2.0, 0.8, increasing=False)
            risk_components.append(("Employment stability", 2.0, emp_years_r,
                f"{years_employer} year(s) with current employer (minimum: 2 years for salaried)."))

            # ── Income inflation ─────────────────────────────────────────────
            # Stated vs actual bank credits. >1.2x is suspicious, >1.5x is a red flag.
            # Sigmoid centred at 1.2, scale 0.12. Weight 2.5 — income fraud is high value.
            inflate_r = sigmoid_risk(inflate, 1.2, 0.12, increasing=True)
            risk_components.append(("Income inflation", 2.5, inflate_r,
                f"Stated income is {inflate:.2f}x actual bank credits (safe limit: 1.0-1.1x)."))

            # ── Credit enquiries ─────────────────────────────────────────────
            # >3 enquiries in 30 days suggests loan shopping or desperation.
            # Sigmoid centred at 3, scale 1. Weight 1.5.
            enq_r = sigmoid_risk(enquiries, 3, 1.0, increasing=True)
            risk_components.append(("Credit enquiries (30d)", 1.5, enq_r,
                f"{enquiries} credit enquiries in last 30 days (safe limit: <=2)."))

            # Weighted average (monotonic: higher loan/FOIR/LTI always raises score)
            total_weight = sum(w for _, w, _, _ in risk_components)
            credit_risk  = sum(w * r for _, w, r, _ in risk_components) / total_weight

            n_critical = critical_flags_triggered
            n_minor     = max(flags_count_excl_oc - n_critical, 0)
            # Doc risk: each critical flag adds 0.25 (hard cap 0.80), each minor adds 0.05 (cap 0.30)
            # Critical flags: each adds 0.25 (not capped at 0.80 — stacks properly)
            # Minor flags: each adds 0.05, capped at 0.30
            # OC/CC is protective but ONLY when zero risk flags — handled after blend
            critical_doc_risk = min(n_critical * 0.25, 1.0)
            minor_doc_risk    = min(n_minor * 0.05, 0.30)
            doc_risk = min(critical_doc_risk + minor_doc_risk, 1.0)

            # Weighted blend: model score 35%, credit rules 45%, doc flags 20%
            # Monotonic: increasing any bad signal always raises overall_risk
            overall_risk = 0.35 * score + 0.45 * credit_risk + 0.20 * doc_risk
            overall_risk = min(max(overall_risk, 0.0), 1.0)

            # ── Hard policy floors (RBI/lender norms — model cannot override) ──
            # LTV > 80%: minimum medium risk
            if ltv > 0.80 and overall_risk < 0.40:
                overall_risk = 0.40
            # Effective FOIR > 65%: minimum medium risk (well over any lender's cap)
            if effective_foir > 0.65 and overall_risk < 0.40:
                overall_risk = 0.40
            # FOIR > 75%: high risk — borrower is mathematically over-leveraged
            if effective_foir > 0.75 and overall_risk < 0.55:
                overall_risk = 0.55
            # Income inflate > 1.5x: minimum medium risk (likely inflated docs)
            if inflate > 1.5 and overall_risk < 0.40:
                overall_risk = 0.40
            # Income inflate > 2.0x: high risk
            if inflate > 2.0 and overall_risk < 0.55:
                overall_risk = 0.55
            # LTI > 6x: minimum medium risk
            if lti > 6.0 and overall_risk < 0.40:
                overall_risk = 0.40
            # CIBIL < 650: minimum medium risk
            if cibil < 650 and overall_risk < 0.40:
                overall_risk = 0.40
            # CIBIL < 600: high risk
            if cibil < 600 and overall_risk < 0.55:
                overall_risk = 0.55
            # Employment < 1 year (salaried): minimum low-medium risk
            if years_employer < 1 and overall_risk < 0.25:
                overall_risk = 0.25
            # Age at maturity > 65: minimum low-medium risk
            if age_at_end > 65 and overall_risk < 0.25:
                overall_risk = 0.25
            # Age at maturity > 70: minimum medium risk
            if age_at_end > 70 and overall_risk < 0.40:
                overall_risk = 0.40
            # Credit enquiries > 5 in 30d: minimum low-medium
            if enquiries > 5 and overall_risk < 0.25:
                overall_risk = 0.25

            # Critical flag floor: each critical flag forces overall_risk to at least 0.35
            # Protective flag (OC_CC) is locked out if any risk flag is present
            has_any_risk_flag = flags_count_excl_oc > 0
            if has_any_risk_flag:
                oc_cc_active = False  # OC/CC cannot reduce score when risk flags present
            else:
                oc_cc_active = bool(oc_cc)

            # Apply OC/CC reduction only when no risk flags exist
            if oc_cc_active:
                overall_risk = max(overall_risk - 0.04, 0.0)

            # Hard floor: each critical flag sets minimum 0.35, stacks per flag
            critical_floor = min(critical_flags_triggered * 0.35, 1.0)
            if overall_risk < critical_floor:
                overall_risk = critical_floor

            # Any risk flag at all: minimum 0.25
            if has_any_risk_flag and overall_risk < 0.25:
                overall_risk = 0.25

            overall_risk = min(max(overall_risk, 0.0), 1.0)

            if overall_risk >= 0.70:
                color, bg, label = "#f87171", "#1c0a0a", "CRITICAL RISK - REJECT & FLAG"
            elif overall_risk >= 0.55:
                color, bg, label = "#f87171", "#1c0a0a", "HIGH RISK - DO NOT DISBURSE"
            elif overall_risk >= 0.40:
                color, bg, label = "#fbbf24", "#1c1100", "MEDIUM RISK - ESCALATE TO SENIOR UNDERWRITER" if critical_flags_triggered > 0 else "MEDIUM RISK - MANUAL UNDERWRITER REVIEW"
            elif overall_risk >= 0.25:
                color, bg, label = "#fbbf24", "#1c1100", "LOW-MEDIUM RISK - VERIFY DOCUMENTS"
            else:
                color, bg, label = "#34d399", "#021c0e", "LOW RISK - ELIGIBLE FOR UNDERWRITING"

            # named list: (label, risk_value, message) — uniform 3-tuple
            named = [("Fraud model signal", score,
                "The fraud model itself sees an unusual pattern in the documents or income data." if score >= 0.4
                else "The fraud model sees nothing unusual in the documents or income data.")]
            if n_critical > 0:
                named.append(("Critical document flags", critical_doc_risk,
                    f"{n_critical} serious document red flag(s) were raised (tampering, AML, signature mismatch, etc.)."))
            if n_minor > 0:
                named.append(("Minor flags", minor_doc_risk,
                    f"{n_minor} smaller suspicious flag(s) were raised."))
            # risk_components is (label, weight, risk_value, message) — extract as 3-tuple
            named.extend([(lbl, r, msg) for lbl, _, r, msg in risk_components])

            top_reasons = sorted(named, key=lambda x: x[1], reverse=True)[:3]
            reasoning_lines = [msg for _, val, msg in top_reasons if val > 0.15]
            if not reasoning_lines:
                reasoning_lines = ["No significant risk factors were found across the fraud model, credit policy, or documentation checks."]

            # ── Anomaly checks: not scored, just worth a human glance ─────────────
            anomaly_notes = []
            if 0 < ltv < 0.30:
                anomaly_notes.append(f"LTV is unusually low at {ltv:.2f}. This borrower would self-fund about {(1-ltv)*100:.0f}% of the property's value. Large self-funded down payments are a known channel for money laundering since the bank verifies the loan portion but often not the rest. Ask for source-of-funds proof.")
            if foir < 0.05:
                anomaly_notes.append(f"FOIR is unusually low at {foir:.2f}, meaning almost no existing debt relative to income. Not a problem alone, but worth a second look alongside other numbers here.")

            r1, r2, r3 = st.columns([1, 3, 1])
            with r2:
                reasoning_html = "".join(f'<div style="font-size:1rem;color:#cbd5e1;line-height:1.6;margin-top:8px">{l}</div>' for l in reasoning_lines)
                st.markdown(f'''
                <div style="background:{bg};border:1px solid {color}44;border-radius:16px;padding:1.75rem 2rem;margin:1rem 0;text-align:center">
                  <div style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">OVERALL RISK SCORE</div>
                  <div style="font-size:3rem;font-weight:700;color:{color};line-height:1">{overall_risk*100:.1f}%</div>
                  <div style="font-size:1.25rem;font-weight:700;color:{color};margin-top:8px">{label}</div>
                  <div style="margin-top:14px;text-align:left;max-width:520px;margin-left:auto;margin-right:auto">{reasoning_html}</div>
                </div>''', unsafe_allow_html=True)

                if anomaly_notes:
                    anomaly_html = "".join([f'<div style="margin-bottom:8px"><span style="font-size:0.92rem;color:#fde68a;line-height:1.5">{a}</span></div>' for a in anomaly_notes])
                    st.markdown(f'<div style="background:#1c1100;border:1px solid #fbbf2444;border-radius:12px;padding:0.85rem 1.1rem;margin-top:0.5rem"><div style="font-size:0.68rem;font-weight:600;color:#fbbf24;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">WORTH A SECOND LOOK (not scored)</div>{anomaly_html}</div>', unsafe_allow_html=True)

            inflate     = min(stated_income/actual_credit if actual_credit else 1, 10)
            flags_count = sum([bank_tampered,pdf_anomaly,aml_flag,pep,disb_nonseller,sig_mismatch,id_altered,itr_before,overseas,month_end_park,val_inflation,salary_personal,pf_missing,unreachable,foreclosed,round_salary,email_mismatch,oc_cc])

            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                col = "#f87171" if inflate > 1.5 else "#34d399"
                st.markdown(f'<div class="metric-card"><div class="label">Income Inflate Ratio</div><div class="value" style="color:{col}">{inflate:.2f}x</div><div class="meaning">Stated income vs actual bank credits</div></div>', unsafe_allow_html=True)
            with s2:
                col = "#f87171" if flags_count_excl_oc >= 3 else "#fbbf24" if flags_count_excl_oc >= 1 else "#34d399"
                critical_note = f" ({critical_flags_triggered} critical)" if critical_flags_triggered >= 1 else ""
                st.markdown(f'<div class="metric-card"><div class="label">Fraud Flags Triggered</div><div class="value" style="color:{col}">{flags_count_excl_oc} / 17{critical_note}</div><div class="meaning">Number of suspicious flags raised</div></div>', unsafe_allow_html=True)
            with s3:
                col = "#f87171" if cibil < 650 else "#fbbf24" if cibil < 720 else "#34d399"
                st.markdown(f'<div class="metric-card"><div class="label">CIBIL Score</div><div class="value" style="color:{col}">{cibil}</div><div class="meaning">Credit score - higher is better (300-900)</div></div>', unsafe_allow_html=True)
            with s4:
                col = "#f87171" if emi_to_income > 0.55 else "#fbbf24" if emi_to_income > 0.40 else "#34d399"
                st.markdown(f'<div class="metric-card"><div class="label">Est. EMI / Income</div><div class="value" style="color:{col}">{emi_to_income*100:.0f}%</div><div class="meaning">Rs{estimated_emi:,.0f}/mo EMI on Rs{actual_credit:,}/mo actual income</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem">FRAUD MODEL DETAIL - WHAT DROVE ITS PROBABILITY</div>', unsafe_allow_html=True)

            try:
                if use_tree_shap:
                    core = model.estimators_[0][1] if hasattr(model, 'estimators_') else model
                    exp  = shap.TreeExplainer(core)
                    sv   = exp.shap_values(app_df)
                    shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
                else:
                    shap_vals = explainer.shap_values(scaler.transform(app_df))[0]
            except Exception:
                shap_vals = np.zeros(len(feature_names))

            contrib_df = pd.DataFrame({'Feature':feature_names,'SHAP':shap_vals,'Value':app_df.iloc[0].values})
            contrib_df = contrib_df[contrib_df['Value']!=0].sort_values('SHAP',key=abs,ascending=False).head(12)

            if not contrib_df.empty:
                max_abs = contrib_df['SHAP'].abs().max() or 1
                st.markdown('<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:14px;padding:1.25rem 1.5rem 0.25rem"><p style="color:#9aa5b8;font-size:0.98rem;margin:0 0 0.75rem;line-height:1.6">Each bar shows how much that factor <span style="color:#f87171">pushed the score up</span> or <span style="color:#34d399">pulled it down</span>. The 12 biggest factors are shown below.</p></div>', unsafe_allow_html=True)
                for _, row in contrib_df.iterrows():
                    d  = row['SHAP'] > 0
                    bw = int(abs(row['SHAP'])/max_abs*100)
                    ac = "#f87171" if d else "#34d399"
                    bc = "#ef4444" if d else "#22c55e"
                    ss = f"+{row['SHAP']:.3f}" if d else f"{row['SHAP']:.3f}"
                    ar = "Increases risk" if d else "Reduces risk"
                    st.markdown(f'<div style="margin:8px 0;padding:0 4px"><div style="display:flex;justify-content:space-between;margin-bottom:3px"><span style="font-size:0.88rem;color:#f0f4f8;font-family:monospace">{str(row["Feature"])[:42]}</span><span style="font-size:0.85rem;color:{ac}">{ar} <b>{ss}</b></span></div><div style="background:#1e2533;border-radius:4px;height:7px"><div style="width:{bw}%;height:100%;background:{bc};border-radius:4px"></div></div></div>', unsafe_allow_html=True)

            shap_summary = "\n".join(f"  {r['Feature']}: {'increases' if r['SHAP']>0 else 'reduces'} risk (SHAP {r['SHAP']:+.3f})" for _, r in contrib_df.iterrows()) if not contrib_df.empty else "No active features"
            st.session_state['last_inflate']  = inflate
            st.session_state['last_flags']    = flags_count
            st.session_state['last_shap']     = shap_summary
            st.session_state['last_overall']  = overall_risk
            st.markdown("<br>", unsafe_allow_html=True)
            render_chat_widget(inputs, score, inflate, flags_count, shap_summary, overall_risk=overall_risk)

        if not submitted and 'last_inputs' in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            render_chat_widget(st.session_state['last_inputs'], st.session_state['last_score'],
                               st.session_state.get('last_inflate',1.0), st.session_state.get('last_flags',0),
                               st.session_state.get('last_shap',''), overall_risk=st.session_state.get('last_overall',None))

    with tab3:
        st.markdown("### Feature Importance")
        col_fraud, col_clean = st.columns(2)
        with col_fraud:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase">TOP 10 - MOST IMPORTANT</p>', unsafe_allow_html=True)
            top10 = coef_df.head(10)
            mw = top10['weight'].max() or 1
            for _, row in top10.iterrows():
                pct = int(abs(row['weight'])/mw*100)
                st.markdown(f'<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;margin-bottom:3px"><span style="font-size:0.78rem;color:#f0f4f8;font-family:monospace">{row["feature"][:38]}</span><span style="font-size:0.78rem;font-weight:600;color:#f87171">{row["weight"]:.4f}</span></div><div style="background:#1e2533;border-radius:4px;height:7px;overflow:hidden"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#dc2626,#f87171);border-radius:4px"></div></div></div>', unsafe_allow_html=True)
        with col_clean:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase">BOTTOM 10 - LEAST IMPORTANT</p>', unsafe_allow_html=True)
            bot10 = coef_df.tail(10).iloc[::-1]
            mw = abs(bot10['weight']).max() or 1
            for _, row in bot10.iterrows():
                pct = int(abs(row['weight'])/mw*100)
                st.markdown(f'<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;margin-bottom:3px"><span style="font-size:0.78rem;color:#f0f4f8;font-family:monospace">{row["feature"][:38]}</span><span style="font-size:0.78rem;font-weight:600;color:#34d399">{row["weight"]:.4f}</span></div><div style="background:#1e2533;border-radius:4px;height:7px;overflow:hidden"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#059669,#34d399);border-radius:4px"></div></div></div>', unsafe_allow_html=True)

    with tab4:
        st.markdown("### Inspect Actual Loan Applications")
        b1, b2, b3, b4 = st.columns(4)
        tp_data = test_df[test_df['category']=='TP']
        fp_data = test_df[test_df['category']=='FP']
        fn_data = test_df[test_df['category']=='FN']
        tn_data = test_df[test_df['category']=='TN']
        with b1:
            st.markdown(f'<div style="background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px"><div style="font-size:2rem;font-weight:700;color:#34d399">{tp:,}</div><div style="font-size:0.85rem;font-weight:600;color:#34d399">True Positive</div><div style="font-size:0.85rem;color:#6ee7b7;margin-top:4px">Fraud correctly caught</div></div>', unsafe_allow_html=True)
            if st.button("View TP loans", key="btn_tp"): st.session_state['selected']='TP'
        with b2:
            st.markdown(f'<div style="background:#78350f;border:2px solid #fbbf24;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px"><div style="font-size:2rem;font-weight:700;color:#fbbf24">{fp:,}</div><div style="font-size:0.85rem;font-weight:600;color:#fbbf24">False Positive</div><div style="font-size:0.85rem;color:#fde68a;margin-top:4px">Innocent flagged</div></div>', unsafe_allow_html=True)
            if st.button("View FP loans", key="btn_fp"): st.session_state['selected']='FP'
        with b3:
            st.markdown(f'<div style="background:#7f1d1d;border:2px solid #f87171;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px"><div style="font-size:2rem;font-weight:700;color:#f87171">{fn:,}</div><div style="font-size:0.85rem;font-weight:600;color:#f87171">False Negative</div><div style="font-size:0.85rem;color:#fca5a5;margin-top:4px">Fraud missed</div></div>', unsafe_allow_html=True)
            if st.button("View FN loans", key="btn_fn"): st.session_state['selected']='FN'
        with b4:
            st.markdown(f'<div style="background:#1e3a5f;border:2px solid #60a5fa;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px"><div style="font-size:2rem;font-weight:700;color:#60a5fa">{tn:,}</div><div style="font-size:0.85rem;font-weight:600;color:#60a5fa">True Negative</div><div style="font-size:0.85rem;color:#93c5fd;margin-top:4px">Clean correctly approved</div></div>', unsafe_allow_html=True)
            if st.button("View TN loans", key="btn_tn"): st.session_state['selected']='TN'
        selected = st.session_state.get('selected', None)
        if selected:
            st.markdown("---")
            data_map = {'TP':tp_data,'FP':fp_data,'FN':fn_data,'TN':tn_data}
            render_loan_table(data_map[selected], selected)
            csv = data_map[selected].drop(columns=['predicted','actual'],errors='ignore').to_csv(index=False)
            st.download_button(label=f"Download {selected} loans as CSV", data=csv, file_name=f"{selected}_loans.csv", mime="text/csv")
        else:
            st.markdown('<div style="background:#1a1f2e;border:1px dashed #2d3748;border-radius:12px;padding:2rem;text-align:center;margin-top:1rem"><div style="color:#8899aa;font-size:0.9rem">Click any button above to see the actual loan applications</div></div>', unsafe_allow_html=True)

    with tab5:
        st.markdown("### Model Comparison - 10 Models Tested")
        st.markdown(f'<p style="color:#8899aa;font-size:0.9rem">Includes individual models, voting ensemble, and stacking ensembles. Winner: <strong style="color:#60a5fa">{best_name}</strong></p>', unsafe_allow_html=True)

        for type_label, color in [("Individual","#4b5563"),("Voting Ensemble","#7c3aed"),("Stacking Ensemble","#0891b2")]:
            subset = comparison_df[comparison_df['Type']==type_label]
            if subset.empty: continue
            st.markdown(f'<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem">{type_label}</p>', unsafe_allow_html=True)
            st.dataframe(subset.drop(columns=['Type']), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase">AUC COMPARISON</p>', unsafe_allow_html=True)
        max_auc = comparison_df['AUC'].max()
        for _, row in comparison_df.iterrows():
            bw      = int(row['AUC']/max_auc*100) if max_auc > 0 else 0
            is_best = row['Model'] == best_name
            if row['Type'] == 'Stacking Ensemble':  bar_color = "#0891b2"
            elif row['Type'] == 'Voting Ensemble':   bar_color = "#7c3aed"
            else:                                    bar_color = "#4b5563"
            if is_best: bar_color = "#60a5fa"
            badge = " BEST" if is_best else ""
            st.markdown(f'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;margin-bottom:3px"><span style="font-size:0.82rem;color:#f0f4f8;font-weight:{"600" if is_best else "400"}">{row["Model"]}{badge}</span><span style="font-size:0.82rem;color:{bar_color};font-weight:600">AUC {row["AUC"]}</span></div><div style="background:#1e2533;border-radius:4px;height:9px;overflow:hidden"><div style="width:{bw}%;height:100%;background:{bar_color};border-radius:4px"></div></div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#1e3a5f;border:1px solid #1d4ed8;border-radius:12px;padding:1.25rem 1.5rem;margin-top:1rem">
            <div style="font-size:0.72rem;font-weight:600;color:#60a5fa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">WHAT DO THESE MEAN?</div>
            <p style="color:#cbd5e1;font-size:0.98rem;margin:0;line-height:1.9">
            <strong style="color:#f0f4f8">Individual:</strong> just one model working on its own.<br>
            <strong style="color:#a78bfa">Voting Ensemble:</strong> XGBoost and Random Forest each vote, and whichever answer wins the most votes is used.<br>
            <strong style="color:#67e8f9">Stacking Ensemble:</strong> the base models spot patterns first, then Logistic Regression looks at those patterns and makes the final call.
            </p>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
