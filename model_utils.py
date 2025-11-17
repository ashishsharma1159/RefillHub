
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def load_data(path):
    df = pd.read_csv(path)
    df.fillna(df.mode().iloc[0], inplace=True)
    df_encoded = pd.get_dummies(df, drop_first=True)
    label_col = 'Likely_to_Use_ReFillHub_Yes'
    X = df_encoded.drop(columns=[label_col], errors='ignore')
    y = df_encoded[label_col]
    return df, X, y

def train_models(X, y):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:,1]
        results[name] = {
            "model": model,
            "acc": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "auc": roc_auc_score(y, y_prob)
        }
    return results

def predict_new(model, df, input_data):
    new_df = df.copy()
    new_row = pd.DataFrame([input_data])
    new_df = pd.concat([new_df, new_row], ignore_index=True)
    encoded = pd.get_dummies(new_df, drop_first=True)
    encoded = encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    preds = model.predict(encoded)
    new_df['Prediction'] = preds
    return new_df
