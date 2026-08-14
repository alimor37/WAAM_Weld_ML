import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_excel('Cleaned data.xlsx')

df['Rp0.2_Mpa'] = pd.to_numeric(df['Rp0.2_Mpa'], errors='coerce')
df['Rm_Mpa'] = pd.to_numeric(df['Rm_Mpa'], errors='coerce')

if 'Weld_Method' in df.columns:
    df['SinglePass'] = ((df['No_of_Beads'] == 1) | (
        df['Weld_Method'].astype(str).str.strip().isin(['Rapid Arc', 'Laser-Hybrid']))).astype(int)
else:
    df['SinglePass'] = (df['No_of_Beads'] == 1).astype(int)

df['Base1100'] = df['Base_Material'].astype(str).apply(lambda x: 1 if '1100' in str(x) else 0)

chemistry_features = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'V', 'Nb', 'Cu', 'Al', 'Ti', 'B', 'O', 'N']

for col in chemistry_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

process_features = ['Pcm', 't85_s', 'No_of_Beads', 'Dilution_Ni_pct', 'SinglePass', 'Base1100']

available_chem = [col for col in chemistry_features if col in df.columns]
features_full = process_features + available_chem

loo = LeaveOneOut()


def run_full_model(target_col, model_name, regressor):
    valid_df = df.dropna(subset=process_features + [target_col]).copy()

    X = valid_df[features_full].values
    y = valid_df[target_col].values

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])

    y_pred = cross_val_predict(pipe, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\n=== Full-Chemistry Model ({model_name}) for {target_col} ===")
    print(f"Sample Size (N): {len(valid_df)}")
    print(f"LOOCV R²: {r2:.3f}")
    print(f"LOOCV RMSE: {rmse:.1f} MPa")
    print("=" * 45)

    if model_name == 'Ridge':
        pipe.fit(X, y)
        coefs = pipe.named_steps['regressor'].coef_
        importance = pd.DataFrame({'Feature': features_full, 'Coefficient': coefs})
        importance['Abs_Coef'] = importance['Coefficient'].abs()
        importance = importance.sort_values(by='Abs_Coef', ascending=False)
        print("Top 5 Most Important Features (Standardized):")
        print(importance[['Feature', 'Coefficient']].head(5).to_string(index=False))


run_full_model('Rp0.2_Mpa', 'Ridge', Ridge(alpha=1.0))
run_full_model('Rm_Mpa', 'PLS', PLSRegression(n_components=3))