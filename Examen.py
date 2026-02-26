# =============================================================================
# EXAMEN PRÁCTICO: Ciencia de datos
# Alumno: Cuadras Camacho Carlos Ariel
# =============================================================================

# -----------------------------------------------------------------------------
# PASO 1: IMPORTACIÓN DE LIBRERÍAS (Descomenta lo que necesites)
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PREPROCESAMIENTO ---
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# --- MODELOS ---
# from sklearn.linear_model import LinearRegression      # Predecir valores continuos (Regresión)
# from sklearn.tree import DecisionTreeClassifier        # Predecir categorías (Clasificación - 1 árbol)
# from sklearn.ensemble import RandomForestClassifier    # Predecir categorías (Clasificación - Bosque)
# import statsmodels.formula.api as smf                  # Regresión con reporte estadístico clásico

# --- MÉTRICAS DE EVALUACIÓN ---
# from sklearn.metrics import mean_squared_error, r2_score  # Para Regresión
# from sklearn.metrics import classification_report         # Para Clasificación

# --- VISUALIZACIÓN AVANZADA ---
# from sklearn.tree import plot_tree                     # Para dibujar árboles

# -----------------------------------------------------------------------------
# PASO 2: CARGA Y EXPLORACIÓN BÁSICA
# -----------------------------------------------------------------------------
# df = pd.read_csv("AQUI_VA_EL_NOMBRE_DEL_ARCHIVO.csv")
# print(df.head())  # Un vistazo rápido a las primeras 5 filas para no trabajar a ciegas

# -----------------------------------------------------------------------------
# PASO 3: PREPROCESAMIENTO Y LIMPIEZA
# -----------------------------------------------------------------------------
# A) ¿Hay texto que deba ser número? (Ej: "Malo", "Bueno") -> Activa el LabelEncoder
# le = LabelEncoder()
# df["Nombre_Columna_Texto"] = le.fit_transform(df["Nombre_Columna_Texto"])

# B) Separar Variables (X = Pistas, y = Respuesta)
# X = df.drop("Nombre_De_La_Columna_A_Predecir", axis=1) 
# y = df["Nombre_De_La_Columna_A_Predecir"]              

# -----------------------------------------------------------------------------
# PASO 4: DIVISIÓN DE DATOS (TRAIN / TEST)
# -----------------------------------------------------------------------------
# Separa el 80% para entrenar y 20% para probar (o 70/30 si el profe lo pide)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------------
# PASO 5: ESCALADO DE DATOS (Opcional: Principalmente para Regresión Lineal)
# -----------------------------------------------------------------------------
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)  # Aprende la regla y transforma
# X_test = scaler.transform(X_test)        # ¡Solo transforma! No uses fit aquí.

# -----------------------------------------------------------------------------
# PASO 6: CREACIÓN Y ENTRENAMIENTO DEL MODELO
# -----------------------------------------------------------------------------
# ELIGE UNO:
# modelo = LinearRegression()                           # Para Regresión Lineal Sklearn
# modelo = DecisionTreeClassifier(max_depth=3)          # Para Árbol de Decisión
# modelo = RandomForestClassifier(n_estimators=100)     # Para Bosque Aleatorio

# modelo.fit(X_train, y_train)  # <--- El modelo aprende aquí (Asegúrate de pasarle X_train_scaled si escalaste)

# -----------------------------------------------------------------------------
# PASO 7: PREDICCIÓN Y BOLETA DE CALIFICACIONES
# -----------------------------------------------------------------------------
# y_pred = modelo.predict(X_test)  # (Pásale X_test_scaled si usaste escalador)

# EVALUACIÓN SEGÚN EL TIPO DE MODELO:

# A) Si es CLASIFICACIÓN (Árboles/Bosques):
# print(classification_report(y_test, y_pred))

# B) Si es REGRESIÓN (Lineal):
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2:", r2_score(y_test, y_pred))

# -----------------------------------------------------------------------------
# PASO 8: VISUALIZACIÓN
# -----------------------------------------------------------------------------
# plt.figure(figsize=(10,6))

# A) Gráfica genérica de dispersión (Regresión):
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') 

# B) Para dibujar tu árbol:
# plot_tree(modelo, feature_names=X.columns, class_names=y.unique().astype(str), filled=True)

# plt.title("Título Descriptivo")
# plt.xlabel("Eje X")
# plt.ylabel("Eje Y")
# plt.grid(True, alpha=0.3)
# plt.show()