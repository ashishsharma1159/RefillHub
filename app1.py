# app1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.metrics import silhouette_score


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_squared_error,
                             r2_score, mean_absolute_error)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Optional (faster boosting)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

st.set_page_config(layout="wide", page_title="ReFillHub - Upgraded Dashboard")

st.title("ReFillHub — Upgraded Marketing & ML Dashboard")
st.markdown("This dashboard includes: **Clustering, Segmentation, Association Mining, 4 Regression models, Decision Tree, Gradient Boosting,** and prediction UI.")

# -------------------------
# Utilities
# -------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df
    
DATA_PATH = "https://raw.githubusercontent.com/ashishsharma1159/RefillHub/main/ReFillHub_SyntheticSurvey.csv"
df = load_data(DATA_PATH)



def auto_detect_columns(df):
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    return numeric_cols, cat_cols

def encode_for_model(df, categorical_cols):
    """One-hot encode categorical columns and return dataframe and encoder info."""
    df_copy = df.copy()
    # Simple label encoding for binary categorical and low-cardinality
    for c in categorical_cols:
        if df_copy[c].nunique() <= 2:
            df_copy[c] = LabelEncoder().fit_transform(df_copy[c].astype(str))
    # One-hot for the rest
    df_copy = pd.get_dummies(df_copy, columns=[c for c in categorical_cols if df_copy[c].nunique() > 2], drop_first=True)
    return df_copy

def train_classifiers(X_train, X_test, y_train, y_test):
    results = {}

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    try:
        auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:,1])
    except Exception:
        auc = np.nan
    results['Decision Tree'] = {
    'model': dt,
    'acc': accuracy_score(y_test, preds),
    'precision': precision_score(y_test, preds, average="weighted", zero_division=0),
    'recall': recall_score(y_test, preds, average="weighted", zero_division=0),
    'f1': f1_score(y_test, preds, average="weighted", zero_division=0),
    'auc': auc
}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    try:
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    except Exception:
        auc = np.nan
    rresults['Random Forest'] = {
    'model': rf,
    'acc': accuracy_score(y_test, preds),
    'precision': precision_score(y_test, preds, average="weighted", zero_division=0),
    'recall': recall_score(y_test, preds, average="weighted", zero_division=0),
    'f1': f1_score(y_test, preds, average="weighted", zero_division=0),
    'auc': auc
}

    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)
    try:
        auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])
    except Exception:
        auc = np.nan


    results['Gradient Boosting'] = {
    'model': gb,
    'acc': accuracy_score(y_test, preds),
    'precision': precision_score(y_test, preds, average="weighted", zero_division=0),
    'recall': recall_score(y_test, preds, average="weighted", zero_division=0),
    'f1': f1_score(y_test, preds, average="weighted", zero_division=0),
    'auc': auc
}


    return results

def train_regressors(X_train, X_test, y_train, y_test):
    results = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    results['Linear Regression'] = {'model': lr, 'RMSE': mean_squared_error(y_test, preds, squared=False),
                                    'MAE': mean_absolute_error(y_test, preds),
                                    'R2': r2_score(y_test, preds)}

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    results['Ridge'] = {'model': ridge, 'RMSE': mean_squared_error(y_test, preds, squared=False),
                        'MAE': mean_absolute_error(y_test, preds), 'R2': r2_score(y_test, preds)}

    lasso = Lasso(max_iter=5000)
    lasso.fit(X_train, y_train)
    preds = lasso.predict(X_test)
    results['Lasso'] = {'model': lasso, 'RMSE': mean_squared_error(y_test, preds, squared=False),
                        'MAE': mean_absolute_error(y_test, preds), 'R2': r2_score(y_test, preds)}

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results['RandomForestRegressor'] = {'model': rf, 'RMSE': mean_squared_error(y_test, preds, squared=False),
                                       'MAE': mean_absolute_error(y_test, preds), 'R2': r2_score(y_test, preds)}

    if XGB_AVAILABLE:
        xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        results['XGBoostRegressor'] = {'model': xgb, 'RMSE': mean_squared_error(y_test, preds, squared=False),
                                      'MAE': mean_absolute_error(y_test, preds), 'R2': r2_score(y_test, preds)}
    return results

# -------------------------
# Load dataset
# -------------------------
st.sidebar.header("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded:
    df = load_data(uploaded)
else:
    st.sidebar.write("No file uploaded. Using a sample file path (update the path below if needed).")
    sample_path = st.sidebar.text_input("Or provide a local CSV path", value="")
    if sample_path:
        df = load_data(sample_path)
    else:
        st.info("Please upload a dataset or give a valid local CSV path in the sidebar.")
        st.stop()

st.sidebar.write("## Dataset overview")
st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.head(200))

numeric_cols, cat_cols = auto_detect_columns(df)

# -------------------------
# Top marketing visuals
# -------------------------
st.header("Top Marketing Insights")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Distribution — Likely_to_Use_ReFillHub")
    if "Likely_to_Use_ReFillHub" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x="Likely_to_Use_ReFillHub", data=df, order=df["Likely_to_Use_ReFillHub"].value_counts().index)
        ax.set_xlabel("")
        st.pyplot(fig)
    else:
        st.write("Column `Likely_to_Use_ReFillHub` not found. Select another column below.")
        sel_col = st.selectbox("Choose column to visualize counts", options=df.columns.tolist())
        fig, ax = plt.subplots(figsize=(6,4))
        try:
            sns.countplot(x=sel_col, data=df, order=df[sel_col].value_counts().index)
            st.pyplot(fig)
        except Exception as e:
            st.write("Could not plot:", e)

with col2:
    st.subheader("Numeric column summaries")
    st.write(df.describe().T)


# -------------------------
# Clustering & Segmentation (with Elbow + Silhouette + Recommended K)
# -------------------------
st.header("Clustering & Segmentation")

seg_features = st.multiselect(
    "Select features for clustering (numeric recommended)",
    options=numeric_cols,
    default=numeric_cols[:4]
)

if seg_features:

    # Prepare scaled data
    X_seg = df[seg_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    # ---------------------------------
    # 1. OPTIMAL CLUSTER FINDER SECTION
    # ---------------------------------
    st.subheader("🔍 Optimal Cluster Finder")

    max_k = st.slider("Maximum number of clusters to test", 4, 12, 8)

    wss = []
    sil = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # ELBOW METRIC (WSS)
        wss.append(km.inertia_)

        # SILHOUETTE METRIC
        if len(set(labels)) > 1:
            sil.append(silhouette_score(X_scaled, labels))
        else:
            sil.append(0)

    # --- Elbow Curve ---
    st.write("### 📉 Elbow Curve (WSS)")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(list(K_range), wss, marker="o")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("WSS")
    ax1.set_title("Elbow Method")
    st.pyplot(fig1)

    # --- Silhouette Curve ---
    st.write("### 🧭 Silhouette Score Curve")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(list(K_range), sil, marker="o", color="orange")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Method")
    st.pyplot(fig2)

    # --- Recommended K ---
    recommended_k = K_range[sil.index(max(sil))]
    st.success(f"Recommended number of clusters (based on silhouette score): **{recommended_k}**")

    # ---------------------------------
    # 2. ACTUAL CLUSTERING SECTION
    # ---------------------------------
    st.subheader("Apply Clustering")

    n_clusters = st.slider("Choose number of clusters to form", 2, max_k, recommended_k)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    seg_df = X_seg.copy()
    seg_df["cluster"] = cluster_labels

    st.write("Cluster Counts")
    st.write(seg_df["cluster"].value_counts())

    # ---------------------------------
    # PCA CLUSTER VISUALIZATION
    # ---------------------------------
    st.subheader("🎨 PCA Visualization of Clusters")

    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(X_scaled)
    plot_df = pd.DataFrame(pca_proj, columns=["PC1", "PC2"])
    plot_df["cluster"] = cluster_labels

    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        x="PC1", y="PC2",
        hue="cluster",
        palette="tab10",
        data=plot_df,
        ax=ax3
    )
    ax3.set_title("PCA 2D Projection of Clusters")
    st.pyplot(fig3)

    # ---------------------------------
    # CLUSTER SUMMARY
    # ---------------------------------
    st.subheader("📊 Cluster Summary (Mean Values)")
    seg_summary = seg_df.groupby("cluster").mean().round(3)
    st.dataframe(seg_summary)


# -------------------------
# Association Mining (Apriori)
# -------------------------
st.header("Association Mining (Apriori)")
if not MLXTEND_AVAILABLE:
    st.warning("mlxtend not installed — install with `pip install mlxtend` to enable Apriori/association rules.")
else:
    # For Apriori we need boolean/onehot transactions. We'll convert selected columns to boolean indicators.
    assoc_cols = st.multiselect("Select categorical columns to use for association rules", options=cat_cols, default=cat_cols[:6])
    min_support = st.slider("Minimum support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.6)

    if assoc_cols:
        trans = pd.DataFrame()
        for c in assoc_cols:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c)
            trans = pd.concat([trans, dummies], axis=1)

        # Run Apriori
        frequent = apriori(trans, min_support=min_support, use_colnames=True)

        if frequent.empty:
            st.write("No frequent itemsets found with this support. Try lowering min_support.")
        else:
            # FIX 1: Convert frozenset → string for Streamlit/Arrow compatibility
            frequent["itemsets"] = frequent["itemsets"].apply(lambda x: ", ".join(list(x)))

            st.write("Top frequent itemsets:")
            st.dataframe(
                frequent.sort_values("support", ascending=False).head(10)
            )

            # Association rules
            rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)

            if rules.empty:
                st.write("No association rules found with this confidence threshold.")
            else:
                st.subheader("Derived association rules")

                # FIX 2: Convert frozenset → string
                rules_display = rules[['antecedents','consequents','support','confidence','lift']].copy()
                rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ", ".join(list(x)))
                rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ", ".join(list(x)))

                st.dataframe(
                    rules_display.sort_values(['confidence','lift'], ascending=False).head(20)
                )



# -------------------------
# Classification Models
# -------------------------
st.header("Classification Models (Decision Tree, RandomForest, GradientBoosting)")

st.write("Choose target for classification (a categorical column).")
class_target = st.selectbox("Classification target", options=[None] + cat_cols, index=0)
if class_target:
    # prepare data
    df_clf = df.dropna(subset=[class_target]).copy()
    # encode target to binary/multiclass numeric with LabelEncoder
    y = LabelEncoder().fit_transform(df_clf[class_target].astype(str))
    X = df_clf.drop(columns=[class_target])
    X_enc = encode_for_model(X, [c for c in X.columns if c in cat_cols])
    # align columns (drop any non-numeric leftovers)
    X_enc = X_enc.select_dtypes(include=[np.number]).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    st.write(f"Training classifiers on {X_train.shape[1]} features...")
    clf_results = train_classifiers(X_train, X_test, y_train, y_test)
    # display results
    res_df = pd.DataFrame({m: {k: v for k, v in r.items() if k!='model'} for m,r in clf_results.items()}).T
    st.dataframe(res_df.round(3))

# -------------------------
# Regression Models
# -------------------------
st.header("Regression Models (Linear, Ridge, Lasso, RandomForest +/- XGBoost if installed)")
st.write("Select a numeric target for regression.")

reg_target = st.selectbox("Regression target", options=[None] + numeric_cols, index=0)
if reg_target:
    df_reg = df.dropna(subset=[reg_target]).copy()
    y = df_reg[reg_target].astype(float)
    X = df_reg.drop(columns=[reg_target])
    X_enc = encode_for_model(X, [c for c in X.columns if c in cat_cols])
    X_enc = X_enc.select_dtypes(include=[np.number]).fillna(0)
    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    st.write("Training regressors...")
    reg_results = train_regressors(X_train, X_test, y_train, y_test)
    # format results
    reg_display = {}
    for name, info in reg_results.items():
        reg_display[name] = {k: round(v,4) for k,v in info.items() if k != 'model'}
    st.dataframe(pd.DataFrame(reg_display).T)

# -------------------------
# Predict new customer (classification & regression)
# -------------------------
st.header("Predict New Customer / New Sample")
st.write("You can provide a new sample to predict classification or regression outcome using the trained models.")

# We'll create inputs for all columns present in df and try to encode similarly.
with st.form("predict_form"):
    st.subheader("Enter values for each column (leave blank to use mode/median)")
    new_sample = {}
    for col in df.columns:
        # sensible defaults
        if col in numeric_cols:
            new_sample[col] = st.text_input(col, value="")
        else:
            new_sample[col] = st.text_input(col, value="")

    submit = st.form_submit_button("Run Predictions")

if submit:
    # Build a DataFrame for new sample
    sample_df = pd.DataFrame([new_sample])
    # coerce numerics
    for c in numeric_cols:
        if c in sample_df.columns:
            val = sample_df.at[0,c]
            try:
                sample_df.at[0,c] = float(val) if val != "" else np.nan
            except:
                sample_df.at[0,c] = np.nan
    # fill blanks with mode/median from original df
    for c in sample_df.columns:
        if sample_df[c].isna().all() or sample_df.at[0,c] in ("", None, np.nan):
            if c in numeric_cols:
                sample_df.at[0,c] = df[c].median()
            else:
                sample_df.at[0,c] = df[c].mode().iloc[0] if not df[c].mode().empty else ""
    st.write("Sample after filling missing with dataset stats:")
    st.write(sample_df.T)

    # Classification prediction if models exist
    if 'clf_results' in locals():
        X_sample = encode_for_model(sample_df.drop(columns=[class_target]) if class_target in df.columns else sample_df, [c for c in df.columns if c in cat_cols])
        # align columns with training data
        X_sample = X_sample.reindex(columns=X_enc.columns, fill_value=0) if 'X_enc' in locals() else X_sample
        preds = {}
        for name, info in clf_results.items():
            model = info['model']
            try:
                p = model.predict(X_sample)
                preds[name] = p[0]
            except Exception as e:
                preds[name] = f"Error: {e}"
        st.subheader("Classification predictions")
        st.write(pd.Series(preds))

    # Regression prediction if models exist
    if 'reg_results' in locals():
        # For regressors we need to encode & scale same as training
        Xs = encode_for_model(sample_df.drop(columns=[reg_target]) if reg_target in df.columns else sample_df, [c for c in df.columns if c in cat_cols])
        # align columns
        if 'X_enc' in locals():
            Xs = Xs.reindex(columns=pd.DataFrame(X_enc, columns=df_reg.drop(columns=[reg_target]).columns if isinstance(X_enc, np.ndarray) else X_enc.columns).columns, fill_value=0)
        # scale with scaler used earlier (if present)
        try:
            Xs_scaled = scaler.transform(Xs.fillna(0))
        except Exception:
            Xs_scaled = Xs.fillna(0).values
        preds_reg = {}
        for name, info in reg_results.items():
            model = info['model']
            try:
                p = model.predict(Xs_scaled)
                preds_reg[name] = float(np.round(p[0], 4))
            except Exception as e:
                preds_reg[name] = f"Error: {e}"
        st.subheader("Regression predictions")
        st.write(pd.Series(preds_reg))

# -------------------------
# Export models/results
# -------------------------
st.header("Export / Save")
if st.button("Save latest cluster summary to CSV"):
    out = seg_summary.reset_index().to_csv(index=False)
    st.download_button("Download cluster summary CSV", data=out, file_name="cluster_summary.csv")

st.write("Notes:")
st.markdown("""
- Clustering uses KMeans on selected numeric features and visualizes clusters via PCA projection.  
- Association Mining requires `mlxtend` and works on selected categorical columns.  
- Classification section trains DecisionTree, RandomForest and GradientBoosting on a chosen categorical target.  
- Regression section trains Linear, Ridge, Lasso and RandomForest regressors (XGBoost if installed) on a numeric target.  
- Predictions attempt to encode your manual sample to the same feature space as training data.  
- If a model shows perfect scores (1.0), watch out for overfitting or data leakage — validate with cross-validation.
""")
