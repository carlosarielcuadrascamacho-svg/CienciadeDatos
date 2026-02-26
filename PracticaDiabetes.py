# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Plantilla optimizada para "Clasificación Binaria Numérica" usando Bosques Aleatorios.
# Ideal para obtener mayor precisión y un mejor 'Recall' en problemas complejos.
# =============================================================================

# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd 
from sklearn.model_selection import train_test_split 

# ¡CAMBIO 1: Importamos el Bosque Aleatorio en lugar del Árbol!
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import classification_report 
from sklearn.tree import plot_tree 
import matplotlib.pyplot as plt 

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
df = pd.read_csv("Documentos/diabetes.csv")

print("--- PRIMERAS 5 FILAS DEL DATASET ---")
print(df.head(), "\n")

# =============================================================================
# 3. DEFINICIÓN DE VARIABLES
# =============================================================================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =============================================================================
# 4. DIVISIÓN DE DATOS
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# =============================================================================
# 5. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# =============================================================================
# ¡CAMBIO 2: Instanciamos RandomForestClassifier y agregamos n_estimators!
# n_estimators=100 significa que crearemos 100 árboles votando al mismo tiempo.
modelo = RandomForestClassifier(n_estimators=10000, random_state=40)
modelo.fit(X_train, y_train)

# =============================================================================
# 6. PREDICCIÓN Y EVALUACIÓN
# =============================================================================
y_pred = modelo.predict(X_test)

print("--- BOLETA DE CALIFICACIONES (REPORTE DE CLASIFICACIÓN) ---")
print(classification_report(y_test, y_pred))

# =============================================================================
# 7. VISUALIZACIÓN (Un solo árbol)
# =============================================================================

# =============================================================================
# EXTRA DE EXAMEN: IMPORTANCIA DE VARIABLES
# =============================================================================
print("\n--- IMPORTANCIA DE LAS VARIABLES EN EL BOSQUE ---")
importancias = modelo.feature_importances_
for columna, importancia in zip(X.columns, importancias):
    print(f"{columna}: {importancia * 100:.2f}%")