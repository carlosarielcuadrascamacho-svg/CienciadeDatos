# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este código implementa un "Bosque Aleatorio" (Random Forest) para CLASIFICACIÓN.
# Úsalo en el examen cuando:
# 1. Tu modelo de un solo Árbol de Decisión no sea lo suficientemente preciso o 
#    tenga problemas de sobreajuste (overfitting).
# 2. Quieras usar la "sabiduría de las multitudes": Random Forest crea MUCHOS árboles 
#    y los hace "votar" para tomar la decisión final. Es mucho más robusto.
# 3. Te pidan visualizar específicamente "uno de los árboles" dentro del bosque.
# =============================================================================

import pandas as pd # Para manejar las tablas de datos
import matplotlib.pyplot as plt # Para dibujar gráficas
from sklearn.model_selection import train_test_split # Para dividir los datos (Train/Test)
from sklearn.preprocessing import LabelEncoder # Para traducir categorías de texto a números

# ¡NUEVA LIBRERÍA ESTRELLA! El algoritmo de Random Forest
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import classification_report # Para el reporte de calificaciones (precision, recall)
from sklearn.tree import plot_tree # Para dibujar el árbol

# 1. CARGA DE DATOS
# Cargamos el dataset de calidad de agua.
df = pd.read_csv("Documentos/CalidadAgua.csv")

# 2. PREPROCESAMIENTO (CODIFICACIÓN)
# Típico de exámenes: Convertimos el texto ("Río", "Red", "Pozo") en números (0, 1, 2).
le_Fuente = LabelEncoder()
le_Temporada = LabelEncoder()
le_Apta = LabelEncoder()

# Aplicamos la transformación directamente a las columnas
df["Fuente"] = le_Fuente.fit_transform(df["Fuente"])
df["Temporada"] = le_Temporada.fit_transform(df["Temporada"])
df["Apta"] = le_Apta.fit_transform(df["Apta"])

# 3. SEPARAR CARACTERÍSTICAS (X) Y OBJETIVO (y)
# X son las pistas que usamos para adivinar. Quitamos la respuesta ("Apta").
X = df.drop("Apta", axis=1)
# y es la respuesta que queremos que la máquina aprenda a adivinar.
y = df["Apta"]

# 4. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# 70% para estudiar (train), 30% para el examen (test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# 5. CREAR Y ENTRENAR EL MODELO (¡PREGUNTA DE EXAMEN!)
# n_estimators=100 significa que el modelo va a crear 100 árboles de decisión diferentes.
# max_depth=3 significa que ninguno de esos 100 árboles puede tener más de 3 niveles de profundidad.
# Cuando prediga, los 100 árboles votarán y ganará la decisión de la mayoría.
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=40)

# .fit() pone a estudiar a los 100 árboles al mismo tiempo con los datos de entrenamiento.
modelo_rf.fit(X_train, y_train)

# 6. PREDICCIONES Y EVALUACIÓN
# Hacemos que el bosque entero prediga la calidad del agua del set de pruebas.
rf_pred = modelo_rf.predict(X_test)

# Imprimimos el reporte para ver si esos 100 árboles juntos lo hicieron mejor que uno solo.
print(classification_report(y_test, rf_pred))

# 7. VISUALIZAR UN ÁRBOL DEL BOSQUE
plt.figure(figsize=(12, 6))

# =============================================================================
# EXTRA DE EXAMEN: ¿Qué variables fueron las más importantes para el Bosque?
# =============================================================================
# Extraemos el porcentaje de importancia que el bosque le dio a cada columna
importancias = modelo_rf.feature_importances_

print("--- IMPORTANCIA DE LAS VARIABLES EN EL BOSQUE ---")
# Hacemos un ciclo para imprimir el nombre de la columna y su nivel de importancia
for columna, importancia in zip(X.columns, importancias):
    # Lo multiplicamos por 100 para verlo como porcentaje
    print(f"{columna}: {importancia * 100:.2f}%")

# ¡OJO AQUÍ EN EL EXAMEN! No puedes dibujar un bosque entero de 100 árboles a la vez.
# modelo_rf.estimators_[0] le dice a Python: "De los 100 árboles que creaste, 
# saca solo el primero (el de la posición 0) y dibuja ese".
plot_tree(modelo_rf.estimators_[0], 
          feature_names=X.columns, 
          class_names=le_Apta.classes_.astype(str), # Convertimos los nombres a string para evitar errores visuales
          filled=True, # Rellena de color las cajitas
          fontsize=10) # Ajusta el tamaño de la letra dentro de los cuadros

# Muestra la ventana con el árbol
plt.show()

