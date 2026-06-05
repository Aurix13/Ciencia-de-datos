import numpy as np
import pandas as pd

# Semilla para reproducir resultados
np.random.seed(42)

# Cantidad de registros
n = 500

# Temperatura simulada
temperatura = np.random.normal(80, 5, n)

# Tasa de error relacionada con la temperatura
tasa_error = (
    temperatura * 0.15
    + np.random.normal(0, 2, n)
)

# Turnos de trabajo
turno = np.random.choice(
    ["Matutino", "Vespertino"],
    n
)

# Crear DataFrame
df = pd.DataFrame({
    "temperatura": temperatura,
    "tasa_error": tasa_error,
    "turno": turno
})

print(df.head())

print("\n=== ESTADÍSTICAS DE TEMPERATURA ===")

print("Media:", df["temperatura"].mean())

print("Mediana:", df["temperatura"].median())

print("Desviación estándar:", df["temperatura"].std())

print("\n=== CREANDO VALORES FALTANTES ===")

# Crear algunos valores faltantes
df.loc[0:10, "temperatura"] = np.nan

print(df.head(12))

print("\n=== VALORES FALTANTES ===")

print(df.isnull().sum())

print("\n=== LIMPIEZA DE DATOS ===")

df["temperatura"] = df["temperatura"].fillna(
    df["temperatura"].mean()
)

print(df.isnull().sum())
print("\n=== CORRELACIÓN DE PEARSON ===")

correlacion = df["temperatura"].corr(
    df["tasa_error"]
)

print("Correlación:", correlacion)
print("\n=== PROMEDIO DE ERRORES POR TURNO ===")

print(
    df.groupby("turno")["tasa_error"].mean()
)
print("\n=== TEMPERATURA PROMEDIO POR TURNO ===")

print(
    df.groupby("turno")["temperatura"].mean()
)
from scipy.stats import ttest_ind

matutino = df[
    df["turno"] == "Matutino"
]["temperatura"]

vespertino = df[
    df["turno"] == "Vespertino"
]["temperatura"]

t, p = ttest_ind(
    matutino,
    vespertino
)

print("\n=== T-TEST ===")

print("Estadístico t:", t)
print("Valor p:", p)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# BOXPLOT
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="turno",
    y="tasa_error",
    data=df
)

plt.title("Tasa de Error por Turno")
plt.xlabel("Turno")
plt.ylabel("Tasa de Error")

plt.show()

# =========================
# SCATTER PLOT
# =========================

plt.figure(figsize=(8,5))

sns.regplot(
    x="temperatura",
    y="tasa_error",
    data=df
)

plt.title("Temperatura vs Tasa de Error")
plt.xlabel("Temperatura")
plt.ylabel("Tasa de Error")

plt.show()