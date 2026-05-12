import pandas as pd
import numpy as np
import os

np.random.seed(42)
DATA = os.path.expanduser('~/Documents/CustomerPulse/data/')

n = 7043
df = pd.DataFrame({
    'customerID': [f'CUS-{i:05d}' for i in range(n)],
    'gender': np.random.choice(['Male','Female'], n),
    'SeniorCitizen': np.random.choice([0,1], n, p=[0.84,0.16]),
    'Partner': np.random.choice(['Yes','No'], n, p=[0.48,0.52]),
    'Dependents': np.random.choice(['Yes','No'], n, p=[0.30,0.70]),
    'tenure': np.random.randint(1, 72, n),
    'PhoneService': np.random.choice(['Yes','No'], n, p=[0.90,0.10]),
    'MultipleLines': np.random.choice(['Yes','No','No phone service'], n, p=[0.42,0.48,0.10]),
    'InternetService': np.random.choice(['DSL','Fiber optic','No'], n, p=[0.34,0.44,0.22]),
    'OnlineSecurity': np.random.choice(['Yes','No','No internet service'], n, p=[0.28,0.50,0.22]),
    'TechSupport': np.random.choice(['Yes','No','No internet service'], n, p=[0.29,0.49,0.22]),
    'StreamingTV': np.random.choice(['Yes','No','No internet service'], n, p=[0.38,0.40,0.22]),
    'Contract': np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.21,0.24]),
    'PaperlessBilling': np.random.choice(['Yes','No'], n, p=[0.59,0.41]),
    'PaymentMethod': np.random.choice([
        'Electronic check','Mailed check',
        'Bank transfer (automatic)','Credit card (automatic)'
    ], n, p=[0.34,0.23,0.22,0.21]),
    'MonthlyCharges': np.round(np.random.uniform(18, 118, n), 2),
})
df['TotalCharges'] = np.round(df['MonthlyCharges'] * df['tenure'] * np.random.uniform(0.95, 1.05, n), 2)
churn_prob = (
    0.05 +
    (df['Contract'] == 'Month-to-month').astype(float) * 0.25 +
    (df['tenure'] < 12).astype(float) * 0.15 +
    (df['InternetService'] == 'Fiber optic').astype(float) * 0.10 +
    (df['SeniorCitizen'] == 1).astype(float) * 0.08 +
    (df['MonthlyCharges'] > 80).astype(float) * 0.08 +
    (df['PaymentMethod'] == 'Electronic check').astype(float) * 0.07 +
    (df['OnlineSecurity'] == 'No').astype(float) * 0.05 +
    (df['TechSupport'] == 'No').astype(float) * 0.05 +
    (df['Partner'] == 'No').astype(float) * 0.03
).clip(0, 1)
df['Churn'] = (np.random.uniform(0, 1, n) < churn_prob).map({True:'Yes', False:'No'})
df.to_csv(DATA + 'customers.csv', index=False)
print(f"Done! {len(df)} customers saved")
print(f"Churn rate: {df['Churn'].value_counts(normalize=True).round(3).to_dict()}")
