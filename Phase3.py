import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_excel('Cleaned data.xlsx')

df['Rp0.2_Mpa'] = pd.to_numeric(df['Rp0.2_Mpa'], errors='coerce')
df['Rm_Mpa'] = pd.to_numeric(df['Rm_Mpa'], errors='coerce')

df['SinglePass'] = ((df['No_of_Beads'] == 1) | (df['Weld_Method'].str.strip().isin(['Rapid Arc', 'Laser-Hybrid']))).astype(int)
df['Base1100'] = df['Base_Material'].astype(str).apply(lambda x: 1 if '1100' in x else 0)

features = ['Pcm', 't85_s', 'No_of_Beads', 'Dilution_Ni_pct', 'SinglePass', 'Base1100']


def evaluate_compact_model(target_col):
    valid_data = df.dropna(subset=features + [target_col]).copy()

    X = valid_data[features].values
    y = valid_data[target_col].values
    model = LinearRegression()
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    r2_loocv = r2_score(y, y_pred)
    rmse_loocv = np.sqrt(mean_squared_error(y, y_pred))
    model.fit(X, y)

    print(f"=== Results for {target_col} ===")
    print(f"Sample Size (N): {len(valid_data)}")
    print(f"LOOCV R²: {r2_loocv:.3f}")
    print(f"LOOCV RMSE: {rmse_loocv:.1f} MPa")
    print("Equation Coefficients:")
    print(f"  Intercept: {model.intercept_:.2f}")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat}: {coef:.2f}")
    print("=" * 35 + "\n")


evaluate_compact_model('Rp0.2_Mpa')
evaluate_compact_model('Rm_Mpa')