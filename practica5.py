import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Semilla
np.random.seed(42)
# Cantidad de registros
n = 300

# Crear DataFrame base
datos_red = pd.DataFrame({
    'duracion_ms': np.random.normal(50, 10, n),
    'paquetes_enviados': np.random.normal(100, 20, n),
    'errores_checksum': np.random.poisson(2, n),
    'latencia_avg': np.random.normal(15, 5, n),
    'jitter': np.random.normal(2, 0.5, n),
    'uso_memoria_sw': np.random.normal(40, 10, n),
    'peticiones_http': np.random.normal(200, 50, n)
})

# Crear dependencias
datos_red['bytes_enviados'] = (
    datos_red['paquetes_enviados'] * 1500
    + np.random.normal(0, 500, n)
)

datos_red['reintentos_tcp'] = (
    datos_red['errores_checksum'] * 1.5
    + np.random.normal(0, 0.5, n)
)

datos_red['carga_cpu_router'] = (
    datos_red['paquetes_enviados'] * 0.4
    + datos_red['latencia_avg'] * 0.2
)

print("=== PRIMERAS FILAS ===")
print(datos_red.head())
print("\n=== MATRIZ DE CORRELACIÓN ===")

correlacion = datos_red.corr()

print(correlacion.round(2))
plt.figure(figsize=(10,8))

sns.heatmap(
    correlacion,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlación entre Variables de Red")
plt.show()
scaler = StandardScaler()

datos_escalados = scaler.fit_transform(
    datos_red
)

print("\n=== DATOS ESTANDARIZADOS ===")
print(datos_escalados[:5])
pca = PCA()

componentes = pca.fit_transform(
    datos_escalados
)

print("\n=== PCA EJECUTADO ===")
print(componentes[:5])
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
    componentes_85 = np.argmax(
    varianza_acumulada >= 0.85
) + 1

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
    "Scree Plot - Tráfico de Red"
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
            len(datos_red.columns)
        )
    ],
    index=datos_red.columns
)

print("\n=== LOADINGS PC1 Y PC2 ===")

print(
    loadings[
        ["PC1", "PC2"]
    ].round(3)
)
plt.figure(figsize=(10,7))

plt.scatter(
    componentes[:,0],
    componentes[:,1]
)

for i, variable in enumerate(
        datos_red.columns):

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
    "Mapa de Estado de Red"
)

plt.grid()

plt.show()