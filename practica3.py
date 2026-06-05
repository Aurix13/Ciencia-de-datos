import numpy as np
import pandas as pd

# Semilla
np.random.seed(42)

# Cantidad de registros
n = 600

# Distancias simuladas
distancia_km = np.random.normal(
    loc=500,
    scale=150,
    size=n
)

# Tipo de transporte
tipo_transporte = np.random.choice(
    ["Terrestre", "Aéreo"],
    size=n
)

# Tiempo de entrega
tiempo_entrega_hrs = []

for transporte, distancia in zip(
        tipo_transporte,
        distancia_km):

    if transporte == "Terrestre":
        tiempo = distancia / 60 + np.random.normal(3, 2)

    else:
        tiempo = distancia / 500 + np.random.normal(2, 1)

    tiempo_entrega_hrs.append(tiempo)

# Costos
costo = []

for transporte, distancia in zip(
        tipo_transporte,
        distancia_km):

    if transporte == "Terrestre":
        costo.append(distancia * 0.8)

    else:
        costo.append(distancia * 2.5)

# DataFrame
df = pd.DataFrame({
    "distancia_km": distancia_km,
    "tiempo_entrega_hrs": tiempo_entrega_hrs,
    "tipo_transporte": tipo_transporte,
    "costo": costo
})

print(df.head())

print("\n=== CREANDO VALORES FALTANTES ===")

df.loc[0:14, "distancia_km"] = np.nan

print(df.head(20))
print("\n=== VALORES FALTANTES ===")

print(df.isnull().sum())

# Imputación por mediana
df["distancia_km"] = df["distancia_km"].fillna(
    df["distancia_km"].median()
)

print("\n=== DESPUÉS DE LIMPIEZA ===")

print(df.isnull().sum())
print("\n=== ESTADÍSTICAS ===")

print(
    "Media:",
    df["tiempo_entrega_hrs"].mean()
)

print(
    "Desviación estándar:",
    df["tiempo_entrega_hrs"].std()
)
print("\n=== CORRELACIÓN DE PEARSON ===")

correlacion = df["distancia_km"].corr(
    df["tiempo_entrega_hrs"]
)

print("Correlación:", correlacion)
print("\n=== PROMEDIOS POR TRANSPORTE ===")

print(
    df.groupby("tipo_transporte")[
        ["tiempo_entrega_hrs", "costo"]
    ].mean()
)
from scipy.stats import ttest_ind

terrestre = df[
    df["tipo_transporte"] == "Terrestre"
]["tiempo_entrega_hrs"]

aereo = df[
    df["tipo_transporte"] == "Aéreo"
]["tiempo_entrega_hrs"]

t, p = ttest_ind(
    terrestre,
    aereo
)

print("\n=== T-TEST ===")

print("Estadístico t:", t)
print("Valor p:", p)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="distancia_km",
    y="tiempo_entrega_hrs",
    hue="tipo_transporte"
)

plt.title(
    "Distancia vs Tiempo de Entrega"
)

plt.xlabel("Distancia (km)")
plt.ylabel("Tiempo de Entrega (hrs)")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="tipo_transporte",
    y="tiempo_entrega_hrs"
)

plt.title(
    "Tiempo de Entrega por Modalidad"
)

plt.xlabel("Tipo de Transporte")
plt.ylabel("Tiempo de Entrega (hrs)")

plt.show()