import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Semilla
np.random.seed(42)

# 500 observaciones
n = 500

# Variable base
base = np.random.normal(100, 10, n)

# Sensores correlacionados
sensor1 = base + np.random.normal(0, 2, n)
sensor2 = base * 1.1 + np.random.normal(0, 3, n)
sensor3 = base * 0.9 + np.random.normal(0, 2, n)
sensor4 = base + np.random.normal(0, 4, n)

sensor5 = np.random.normal(50, 5, n)
sensor6 = sensor5 * 1.2 + np.random.normal(0, 2, n)

sensor7 = np.random.normal(75, 8, n)
sensor8 = sensor7 * 0.8 + np.random.normal(0, 2, n)

sensor9 = np.random.normal(30, 4, n)
sensor10 = np.random.normal(40, 5, n)
sensor11 = np.random.normal(60, 6, n)
sensor12 = np.random.normal(90, 7, n)

df = pd.DataFrame({
    "sensor1": sensor1,
    "sensor2": sensor2,
    "sensor3": sensor3,
    "sensor4": sensor4,
    "sensor5": sensor5,
    "sensor6": sensor6,
    "sensor7": sensor7,
    "sensor8": sensor8,
    "sensor9": sensor9,
    "sensor10": sensor10,
    "sensor11": sensor11,
    "sensor12": sensor12
})
print(df.head())
print("\n=== MATRIZ DE CORRELACIÓN ===")

correlacion = df.corr()

print(correlacion)
plt.figure(figsize=(10,8))

sns.heatmap(
    correlacion,
    annot=False,
    cmap="coolwarm"
)

plt.title(
    "Correlación entre Sensores"
)

plt.show()
scaler = StandardScaler()

datos_escalados = scaler.fit_transform(df)
pca = PCA()

componentes = pca.fit_transform(
    datos_escalados
)
print("\n=== VARIANZA EXPLICADA ===")

print(
    pca.explained_variance_ratio_
)
varianza_acumulada = np.cumsum(
    pca.explained_variance_ratio_
)

print("\n=== VARIANZA ACUMULADA ===")

print(varianza_acumulada)
componentes_85 = np.argmax(
    varianza_acumulada >= 0.85
) + 1

print(
    "\nComponentes necesarios para 85%:",
    componentes_85
)
plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(varianza_acumulada)+1),
    varianza_acumulada,
    marker="o"
)

plt.axhline(
    y=0.85,
    color="red",
    linestyle="--"
)

plt.xlabel("Componentes")

plt.ylabel("Varianza Acumulada")

plt.title("Scree Plot")

plt.show()
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        f"PC{i+1}"
        for i in range(len(df.columns))
    ],
    index=df.columns
)

print("\n=== LOADINGS DEL PC1 ===")

print(
    loadings["PC1"]
    .sort_values(
        ascending=False
    )
)
plt.figure(figsize=(8,6))

plt.scatter(
    componentes[:,0],
    componentes[:,1],
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Biplot PCA")

plt.show()
