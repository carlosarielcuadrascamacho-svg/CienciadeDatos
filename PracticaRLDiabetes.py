# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este código es la "navaja suiza" de la Regresión Lineal en Machine Learning.
# Úsalo en el examen cuando te pidan:
# 1. Predecir un número continuo (como el BMI, Precio, Temperatura) usando múltiples variables.
# 2. Dibujar varias gráficas en una sola imagen (Subplots).
# 3. Evaluar el modelo con métricas de error formales como MSE, RMSE o R² (R-cuadrado).
# 4. Hacer un "Análisis de Residuos", que es básicamente graficar en qué se equivocó 
#    el modelo para ver si hay algún patrón en sus errores.
# =============================================================================

# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd  # Para la manipulación y estructuración de los datos
import matplotlib.pyplot as plt  # Para la generación de gráficos y visualizaciones

from sklearn.linear_model import LinearRegression # Algoritmo de Regresión Lineal
from sklearn.model_selection import train_test_split # Para particionar el dataset (Train/Test)
from sklearn.preprocessing import StandardScaler # Para estandarizar/escalar las escalas numéricas

# LIBRERÍAS NUEVAS (¡Claves para el examen!):
# Para sacar tu "boleta de calificaciones" numérica.
from sklearn.metrics import mean_squared_error, r2_score 
import numpy as np # Para hacer matemáticas, como sacar raíces cuadradas (np.sqrt)

# =============================================================================
# 2. CARGA Y SELECCIÓN DE DATOS
# =============================================================================
df = pd.read_csv("Documentos/diabetes.csv")

# X (Entradas): Variables médicas predictoras. 
# PREGUNTA DE EXAMEN: ¿Por qué quitamos 'Outcome'? Porque no queremos mezclar 
# el diagnóstico final (que es categórico) para predecir el BMI.
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']] 

# y (Objetivo): Lo que queremos predecir, el Índice de Masa Corporal (BMI)
y = df['BMI']

# =============================================================================
# 3. DIVISIÓN DE DATOS
# =============================================================================
# 80% para entrenar y 20% para el examen final.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# 4. ESCALADO DE DATOS (REGLA DE ORO)
# =============================================================================
scaler = StandardScaler()
# Al TRAIN se le aplica fit_transform (aprende la escala y transforma)
X_train_scaled = scaler.fit_transform(X_train)
# Al TEST SOLO se le aplica transform (usa las reglas del train, no hace trampa)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# =============================================================================
modelo = LinearRegression()  # Creamos el modelo
modelo.fit(X_train_scaled, y_train) # Le damos a "Estudiar"

# =============================================================================
# 6. PREDICCIÓN
# =============================================================================
y_pred = modelo.predict(X_test_scaled) # Le hacemos el examen final

# =============================================================================
# 7. VISUALIZACIÓN: GRÁFICAS DE DISPERSIÓN Y RESIDUOS
# =============================================================================
# Crea una figura rectangular grande (12 de ancho x 5 de alto)
plt.figure(figsize=(12, 5))

# --- Subplot 1: Valores Reales vs Predichos ---
# plt.subplot(filas, columnas, posición). (1, 2, 1) = 1 fila, 2 columnas, ponme en el espacio 1.
plt.subplot(1, 2, 1)

# Compara la realidad vs la predicción.
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6, edgecolors='black', linewidth=0.5)

# Línea de la predicción matemáticamente perfecta
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción Perfecta')

plt.xlabel('BMI Real')
plt.ylabel('BMI Predicho')
plt.title('Regresión Lineal: BMI Real vs Predicho')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Subplot 2: Análisis de Residuos (Errores) (¡MUY COMÚN EN EXÁMENES!) ---
# (1, 2, 2) = 1 fila, 2 columnas, ponme en el espacio 2 (a la derecha de la primera gráfica).
plt.subplot(1, 2, 2)

# Residuos = Lo que era en realidad - Lo que predijo el modelo. (Básicamente, por cuánto se equivocó)
residuos = y_test - y_pred

# Dibujamos en X lo que predijo y en Y su error (residuo).
plt.scatter(y_pred, residuos, color='orange', alpha=0.6, edgecolors='black', linewidth=0.5)

# axhline dibuja una línea horizontal en el Cero. 
# Si los puntos naranjas están cerca de la línea roja, el modelo se equivocó muy poco.
plt.axhline(y=0, color='r', linestyle='--', lw=2)

plt.xlabel('BMI Predicho')
plt.ylabel('Residuos (Error)')
plt.title('Análisis de Residuos (BMI)')
plt.grid(True, alpha=0.3)

# tight_layout() es mágico: acomoda las gráficas para que no se encimen los títulos y números.
plt.tight_layout()

# ¡Ojo! Guardamos la foto ANTES del plt.show()
plt.savefig('regresion_lineal_bmi.png')

# =============================================================================
# 8. REPORTE DE EVALUACIÓN Y MÉTRICAS (PREGUNTAS CLÁSICAS)
# =============================================================================
# MSE (Error Cuadrático Medio): Promedio de errores al cuadrado (penaliza errores grandes).
mse = mean_squared_error(y_test, y_pred)

# RMSE (Raíz del Error Cuadrático Medio): Lo mismo que el MSE, pero con raíz cuadrada para 
# que esté en la misma unidad de medida que el BMI original.
rmse = np.sqrt(mse)

# R² (R-cuadrado): Qué tan bueno es el modelo (0 es pésimo, 1 es perfecto).
r2 = r2_score(y_test, y_pred)

print("\n=== REPORTE DE REGRESIÓN LINEAL (BMI) ===")
print(f"Error Cuadrático Medio (MSE): {mse:.6f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.6f}")
print(f"Coeficiente de Determinación (R²): {r2:.6f}")

# Lista para iterar sobre los nombres
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']

# Imprime el impacto de cada variable. Si Glucose tiene 1.5, significa que por cada punto que
# suba la glucosa, el BMI sube 1.5.
print("\n=== IMPORTANCIA DE VARIABLES PARA PREDECIR EL BMI ===")
for i in range(len(cols)):
    print(f"Impacto de {cols[i]}: {modelo.coef_[i]:.4f}")

# El intercepto (bias) es el valor del BMI si TODAS las variables fueran exactamente cero.
print(f"Intersección (bias): {modelo.intercept_:.4f}")