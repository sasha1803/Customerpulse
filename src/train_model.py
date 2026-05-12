import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA = os.path.expanduser('~/Documents/CustomerPulse/data/')

print("Loading and cleaning data...")
df = pd.read_csv(DATA + 'customers.csv')

# Feature engineering
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12m','12-24m','24-48m','48-72m'])
df['high_value'] = (df['MonthlyCharges'] > 80).astype(int)
df['no_support'] = ((df['TechSupport']=='No') & (df['OnlineSecurity']=='No')).astype(int)

# Encode categoricals
le = LabelEncoder()
cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
            'InternetService','OnlineSecurity','TechSupport','StreamingTV',
            'Contract','PaperlessBilling','PaymentMethod','tenure_group']

df_model = df.copy()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

features = ['gender','SeniorCitizen','Partner','Dependents','tenure',
            'PhoneService','MultipleLines','InternetService','OnlineSecurity',
            'TechSupport','StreamingTV','Contract','PaperlessBilling',
            'PaymentMethod','MonthlyCharges','TotalCharges','high_value','no_support']

X = df_model[features]
y = (df_model['Churn'] == 'Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print(f"Logistic Regression — Accuracy: {lr_acc:.3f} | AUC-ROC: {lr_auc:.3f}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"Random Forest — Accuracy: {rf_acc:.3f} | AUC-ROC: {rf_auc:.3f}")

# Save best model and feature info
best_model = rf if rf_auc > lr_auc else lr
best_name = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
print(f"\nBest model: {best_name}")

# Feature importance
feat_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
feat_importance.to_csv(DATA + 'feature_importance.csv', index=False)

# Save model results
results = pd.DataFrame({
    'model': ['Logistic Regression', 'Random Forest'],
    'accuracy': [lr_acc, rf_acc],
    'auc_roc': [lr_auc, rf_auc]
})
results.to_csv(DATA + 'model_results.csv', index=False)

# Save predictions on test set
X_test_copy = X_test.copy()
X_test_copy['actual'] = y_test.values
X_test_copy['rf_prob'] = rf.predict_proba(X_test)[:,1]
X_test_copy['lr_prob'] = lr.predict_proba(X_test)[:,1]
X_test_copy['churn_risk'] = pd.cut(X_test_copy['rf_prob'],
    bins=[0,0.3,0.6,1.0], labels=['Low','Medium','High'])
X_test_copy.to_csv(DATA + 'predictions.csv', index=False)

# Save model
with open(DATA + 'rf_model.pkl', 'wb') as f:
    pickle.dump({'model': rf, 'features': features, 'lr': lr}, f)

print(f"\nTop 5 churn drivers:")
print(feat_importance.head())
print("\nAll files saved!")
