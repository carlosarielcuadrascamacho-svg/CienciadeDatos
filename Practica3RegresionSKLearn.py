# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este es un flujo de trabajo (pipeline) completo de Machine Learning con Scikit-Learn.
# Te sirve en el examen para:
# 1. Aplicar Regresión Lineal Múltiple pero con el enfoque de Machine Learning.
# 2. Dividir tus datos correctamente en "Entrenamiento" y "Prueba" (Train/Test Split) 
#    para comprobar si tu modelo realmente aprendió o solo memorizó.
# 3. Escalar/Normalizar variables (StandardScaler) para que variables con números 
#    grandes (como el riego) no opaquen a variables con números chicos (como temperatura).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# -----------------------------------------------------------------------------
import pandas as pd  # Para manejar tablas de datos
import matplotlib.pyplot as plt  # Para dibujar las gráficas

# Herramientas de Machine Learning (Scikit-Learn):
from sklearn.linear_model import LinearRegression # El algoritmo matemático para predecir
from sklearn.model_selection import train_test_split # Para partir los datos (Train/Test)
from sklearn.preprocessing import StandardScaler # Para nivelar/escalar los datos

# 2. CARGA Y SELECCIÓN DE DATOS
# Cargar el archivo CSV en la variable 'df'.
df = pd.read_csv("Documentos/agricultura.csv")

# DEFINIMOS LAS VARIABLES:
# X (Características/Entradas): Aquello que usas para adivinar. 
# OJO: Lleva DOBLE corchete [['...']] porque es una matriz (varias columnas).
X = df[['Temperatura_C', 'Humedad', 'Riego_mm']] 

# y (Objetivo/Salida): Lo que quieres predecir (El rendimiento).
# OJO: Lleva UN corchete ['...'] porque es un vector (una sola columna).
y = df['Rendimiento_t_ha']

# 3. DIVISIÓN DE DATOS (Típica pregunta de examen)
# Partimos los datos:
# - X_train, y_train (80%): Los apuntes con los que la computadora estudia.
# - X_test, y_test (20%): El examen final que la computadora nunca ha visto.
# random_state=42: Congela la aleatoriedad. Si el profesor corre tu código, 
# se dividirán exactamente los mismos datos que a ti.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ESCALADO DE DATOS
scaler = StandardScaler()

# ¿Por qué se hace? Para que todas las columnas jueguen en las mismas condiciones.
# Evita que el Riego (ej. 500mm) valga más que la Temperatura (ej. 30°C) solo por ser un número mayor.

# REGLA DE ORO EN EXÁMENES DE ML:
# 1. Al X_train se le aplica .fit_transform() porque el modelo debe aprender la escala de aquí.
X_train_scaled = scaler.fit_transform(X_train)

# 2. Al X_test SOLO se le aplica .transform() porque usamos las reglas aprendidas arriba.
# NUNCA uses fit_transform en test, eso es "hacer trampa" (se llama data leakage).
X_test_scaled = scaler.transform(X_test)

# 5. CREACIÓN Y ENTRENAMIENTO DEL MODELO
modelo = LinearRegression()  # Instanciamos (creamos) el modelo vacío.

# .fit() es el botón de "Aprender". 
# Le pasamos las variables X escaladas y las respuestas y.
modelo.fit(X_train_scaled, y_train)

# 6. PREDICCIÓN
# Hora del examen final. Le pasamos al modelo los datos escalados de prueba (X_test_scaled)
# para que nos devuelva sus predicciones ('y_pred').
y_pred = modelo.predict(X_test_scaled)

# 7. VISUALIZACIÓN
plt.figure(figsize=(8, 6)) # Tamaño del lienzo

# Puntos Azules: Eje X (Lo que realmente pasó: y_test) vs Eje Y (Lo que el modelo predijo: y_pred).
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)

# Línea Roja: Representa la perfección (Predicción = Realidad).
# Traza una diagonal desde el valor mínimo hasta el máximo de y.
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 

# Estética de la gráfica
plt.xlabel('Rendimiento Real (t/ha)')
plt.ylabel('Rendimiento Predicho (t/ha)')
plt.title('Comparación: ¿Qué tan cerca estuvo el modelo?')
plt.grid(True) # Activa cuadrícula
plt.show()

# 8. REPORTE DE CALIFICACIÓN (COEFICIENTES)
# modelo.coef_ guarda el "peso" que el modelo le dio a cada variable.
# Entre más lejos esté de 0 (positivo o negativo), mayor es su impacto en el rendimiento.
cols = ['Temperatura', 'Humedad', 'Riego']
for i in range(len(cols)):
    print(f"Impacto de {cols[i]}: {modelo.coef_[i]:.4f}")