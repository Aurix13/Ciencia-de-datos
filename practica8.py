import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

# Semilla
np.random.seed(444)
n = 400

df_red = pd.DataFrame({
    "latencia": np.random.normal(20, 5, n),
    "intentos_fallidos": np.random.poisson(1, n),
    "tamano_paquete": np.random.normal(500, 100, n)
})

df_red["es_ataque"] = np.where(
    (df_red["intentos_fallidos"] > 2) |
    (df_red["latencia"] > 35),
    1,
    0
)

print("=== PRIMERAS FILAS ===")
print(df_red.head())
X = df_red[
    [
        "latencia",
        "intentos_fallidos",
        "tamano_paquete"
    ]
]

y = df_red["es_ataque"]

modelo_log = LogisticRegression(
    max_iter=1000
)

modelo_log.fit(X, y)

print("\n=== REGRESIÓN LOGÍSTICA ===")

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo_log.coef_[0]
})

print(coeficientes)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

resultados = []

for k in range(1, 11):

    modelo_knn = KNeighborsClassifier(
        n_neighbors=k
    )

    accuracy = cross_val_score(
        modelo_knn,
        X_scaled,
        y,
        cv=5,
        scoring="accuracy"
    ).mean()

    resultados.append(
        [k, accuracy]
    )

resultados_knn = pd.DataFrame(
    resultados,
    columns=["k", "accuracy"]
)

print("\n=== RESULTADOS KNN ===")
print(resultados_knn)
plt.figure(figsize=(8,5))

plt.plot(
    resultados_knn["k"],
    resultados_knn["accuracy"],
    marker="o"
)

plt.xlabel("Número de Vecinos (k)")
plt.ylabel("Accuracy")

plt.title(
    "Accuracy de KNN"
)

plt.grid()

plt.show()
mejor_k = resultados_knn.loc[
    resultados_knn["accuracy"].idxmax()
]

print("\n=== MEJOR K ===")
print(mejor_k)
datos_cluster = df_red.drop(
    columns=["es_ataque"]
)

datos_cluster = StandardScaler().fit_transform(
    datos_cluster
)

kmeans = KMeans(
    n_clusters=2,
    random_state=111,
    n_init=10
)

clusters = kmeans.fit_predict(
    datos_cluster
)

df_red["cluster"] = clusters
print("\n=== TABLA COMPARATIVA ===")

tabla = pd.crosstab(
    df_red["es_ataque"],
    df_red["cluster"]
)

print(tabla)
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df_red,
    x="latencia",
    y="intentos_fallidos",
    hue="cluster",
    palette="Set1"
)

plt.title(
    "Agrupamiento de Conexiones"
)

plt.show()