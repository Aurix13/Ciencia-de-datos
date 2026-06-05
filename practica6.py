import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Semilla
np.random.seed(123)
# Cantidad de registros
n = 400

datos_urbanos = pd.DataFrame({
    'temp_ambiente': np.random.normal(28, 4, n),
    'humedad_rel': np.random.normal(60, 10, n),
    'co2_ppm': np.zeros(n),
    'no2_ppb': np.zeros(n),
    'particulas_pm25': np.zeros(n),
    'nivel_ruido_db': np.zeros(n),
    'densidad_vehiculos': np.random.normal(120, 30, n),
    'velocidad_viento': np.random.normal(15, 5, n),
    'radiacion_solar': np.random.normal(800, 100, n),
    'conteo_peatones': np.random.normal(50, 15, n)
})

# Crear correlaciones
datos_urbanos['co2_ppm'] = (
    datos_urbanos['densidad_vehiculos'] * 3.5
    + np.random.normal(300, 50, n)
)

datos_urbanos['no2_ppb'] = (
    datos_urbanos['co2_ppm'] * 0.1
    + np.random.normal(5, 2, n)
)

datos_urbanos['particulas_pm25'] = (
    datos_urbanos['co2_ppm'] * 0.05
    + np.random.normal(10, 3, n)
)

datos_urbanos['nivel_ruido_db'] = (
    datos_urbanos['densidad_vehiculos'] * 0.2
    + 50
    + np.random.normal(0, 5, n)
)

print("=== PRIMERAS FILAS ===")
print(datos_urbanos.head())
correlacion = datos_urbanos.corr()

print("\n=== MATRIZ DE CORRELACIÓN ===")
print(correlacion.round(2))
plt.figure(figsize=(10,8))

sns.heatmap(
    correlacion,
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlación entre Sensores Urbanos"
)

plt.show()
scaler = StandardScaler()

datos_escalados = scaler.fit_transform(
    datos_urbanos
)
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

for i, valor in enumerate(
        varianza_acumulada):

    print(
        f"PC{i+1}: {valor:.4f}"
    )
    componentes_85 = (
    np.argmax(
        varianza_acumulada >= 0.85
    ) + 1
)

print("\n=== COMPONENTES NECESARIOS ===")
print(
    f"Se necesitan {componentes_85} componentes para explicar al menos el 85% de la varianza."
)
plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(
            pca.explained_variance_ratio_
        ) + 1
    ),
    pca.explained_variance_ratio_,
    marker="o"
)

plt.title(
    "Scree Plot - Smart City"
)

plt.xlabel(
    "Componentes Principales"
)

plt.ylabel(
    "Varianza Explicada"
)

plt.grid()

plt.show()
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        f"PC{i+1}"
        for i in range(
            len(datos_urbanos.columns)
        )
    ],
    index=datos_urbanos.columns
)

print("\n=== LOADINGS PC1 Y PC2 ===")

print(
    loadings[
        ["PC1","PC2"]
    ].round(3)
)
plt.figure(figsize=(10,7))

plt.scatter(
    componentes[:,0],
    componentes[:,1]
)

for i, variable in enumerate(
        datos_urbanos.columns):

    plt.arrow(
        0,
        0,
        pca.components_[0,i]*5,
        pca.components_[1,i]*5
    )

    plt.text(
        pca.components_[0,i]*5.2,
        pca.components_[1,i]*5.2,
        variable
    )

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Mapa Biplot de Estado Urbano"
)

plt.grid()

plt.show()
