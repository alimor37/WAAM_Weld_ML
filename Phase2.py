import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_excel('Cleaned data.xlsx')

df['Rp0.2_Mpa'] = pd.to_numeric(df['Rp0.2_Mpa'], errors='coerce')
df['Rm_Mpa'] = pd.to_numeric(df['Rm_Mpa'], errors='coerce')

df['Base_Material'] = df['Base_Material'].astype(str).str.strip()
carbon_equivalents = ['Pcm', 'CE_IIW', 'Cen', 'CET']

palette = {'Weldox 700': '#1f77b4', 'Weldox 1100': '#ff7f0e'}
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(' Carbon Equivalents vs. Yield Strength (Rp0.2)', fontsize=16)


for idx, ce in enumerate(carbon_equivalents):
    ax = axes[idx // 2, idx % 2]

    valid_data = df.dropna(subset=[ce, 'Rp0.2_Mpa', 'Base_Material'])

    X = valid_data[[ce]].values
    y = valid_data['Rp0.2_Mpa'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    sns.scatterplot(data=valid_data, x=ce, y='Rp0.2_Mpa', hue='Base_Material', palette=palette, ax=ax, s=80, alpha=0.8)

    # Trendline
    sort_idx = np.argsort(X.flatten())
    ax.plot(X[sort_idx], y_pred[sort_idx], color='black', linestyle='--', linewidth=2,
            label='Overall Trend' if idx == 0 else "")

    ax.set_title(f'{ce} vs Rp0.2 (Overall R² = {r2:.3f})')
    ax.set_xlabel(ce)
    ax.set_ylabel('Rp0.2 (MPa)')
    ax.grid(True, linestyle=':', alpha=0.6)

    # add legend to thr first plot
    if idx == 0:
        ax.legend(title='Base Material')
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('benchmark_rp02_colored.png', dpi=300)
plt.show()

# draw the second plot
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Carbon Equivalents vs. Tensile Strength (Rm)', fontsize=16)

for idx, ce in enumerate(carbon_equivalents):
    ax = axes2[idx // 2, idx % 2]

    valid_data = df.dropna(subset=[ce, 'Rm_Mpa', 'Base_Material'])

    X = valid_data[[ce]].values
    y = valid_data['Rm_Mpa'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    sns.scatterplot(data=valid_data, x=ce, y='Rm_Mpa', hue='Base_Material', palette=palette, ax=ax, s=80, alpha=0.8)

    sort_idx = np.argsort(X.flatten())
    ax.plot(X[sort_idx], y_pred[sort_idx], color='black', linestyle='--', linewidth=2)

    ax.set_title(f'{ce} vs Rm (Overall R² = {r2:.3f})')
    ax.set_xlabel(ce)
    ax.set_ylabel('Rm (MPa)')
    ax.grid(True, linestyle=':', alpha=0.6)

    if idx == 0:
        ax.legend(title='Base Material')
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('benchmark_rm_colored.png', dpi=300)
plt.show()