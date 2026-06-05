import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score
)

# Semilla
np.random.seed(789)
n = 500

df_empleados = pd.DataFrame({
    "experiencia": np.random.normal(5, 2, n),
    "certificaciones": np.random.poisson(3, n),
    "habilidades_sociales": np.random.uniform(1, 10, n),
    "remoto": np.random.choice([0,1], n)
})

# Salario
df_empleados["salario"] = (
    25000
    + df_empleados["experiencia"] * 5000
    + df_empleados["certificaciones"] * 2000
    + np.random.normal(0, 3000, n)
)

# Retención
prob = 1 / (
    1 + np.exp(
        -(
            -2
            + 0.0001 * df_empleados["salario"]
            + 0.2 * df_empleados["habilidades_sociales"]
        )
    )
)

df_empleados["retencion"] = np.where(
    np.random.rand(n) < prob,
    1,
    0
)

print("=== PRIMERAS FILAS ===")
print(df_empleados.head())
train_data, test_data = train_test_split(
    df_empleados,
    test_size=0.20,
    random_state=123
)
X_train = train_data[
    ["experiencia", "certificaciones"]
]

y_train = train_data["salario"]

X_test = test_data[
    ["experiencia", "certificaciones"]
]

y_test = test_data["salario"]

modelo_salario = LinearRegression()

modelo_salario.fit(
    X_train,
    y_train
)
pred_salario = modelo_salario.predict(
    X_test
)

r2 = r2_score(
    y_test,
    pred_salario
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred_salario
    )
)

print("\n=== REGRESIÓN LINEAL ===")

print(
    "Intercepto:",
    modelo_salario.intercept_
)

print(
    "Coeficientes:",
    modelo_salario.coef_
)

print("R²:", r2)

print("RMSE:", rmse)
X_train_knn = train_data[
    [
        "experiencia",
        "habilidades_sociales",
        "salario"
    ]
]

X_test_knn = test_data[
    [
        "experiencia",
        "habilidades_sociales",
        "salario"
    ]
]

y_train_knn = train_data["retencion"]
y_test_knn = test_data["retencion"]
scaler = StandardScaler()

X_train_knn = scaler.fit_transform(
    X_train_knn
)

X_test_knn = scaler.transform(
    X_test_knn
)

modelo_knn = KNeighborsClassifier(
    n_neighbors=5
)

modelo_knn.fit(
    X_train_knn,
    y_train_knn
)

pred_knn = modelo_knn.predict(
    X_test_knn
)

accuracy = accuracy_score(
    y_test_knn,
    pred_knn
)

print("\n=== KNN ===")
print("Accuracy:", accuracy)
datos_cluster = df_empleados[
    [
        "experiencia",
        "salario"
    ]
]

datos_cluster = StandardScaler().fit_transform(
    datos_cluster
)

kmeans = KMeans(
    n_clusters=3,
    random_state=456,
    n_init=10
)

clusters = kmeans.fit_predict(
    datos_cluster
)

df_empleados["cluster"] = clusters
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df_empleados,
    x="experiencia",
    y="salario",
    hue="cluster",
    palette="Set1"
)

plt.title(
    "Segmentación de Empleados por Perfil"
)

plt.show()
print("\n=== CLUSTERS ===")

print(
    df_empleados.groupby(
        "cluster"
    )[
        [
            "experiencia",
            "salario"
        ]
    ].mean()
)
