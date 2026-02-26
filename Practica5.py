# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Al igual que el anterior, este código es perfecto para problemas de CLASIFICACIÓN
# mediante Árboles de Decisión, pero enfocado en un caso de "Calidad de Agua".
# Úsalo en el examen para:
# 1. Resolver problemas donde la respuesta es categórica (ej. "Sí es apta" o "No es apta").
# 2. Reafirmar el uso de LabelEncoder cuando tienes múltiples columnas de texto
#    (como Fuente del agua o Temporada del año) que deben convertirse a números.
# 3. Dibujar de forma gráfica la lógica que sigue la máquina para tomar una decisión.
# =============================================================================

import pandas as pd # Importa pandas para manejar la tabla de datos (DataFrames)
from sklearn.model_selection import train_test_split # Para dividir los datos en Entrenamiento y Prueba
from sklearn.preprocessing import LabelEncoder # Para convertir texto (Categorías) en números matemáticos
from sklearn.tree import DecisionTreeClassifier # El algoritmo estrella de hoy: Árbol de Decisión
from sklearn.metrics import classification_report # Para generar tu "boleta de calificaciones" (métricas de error/acierto)
from sklearn.tree import plot_tree # Para dibujar el árbol de manera visual
import matplotlib.pyplot as plt # Para configurar el tamaño y mostrar la ventana del gráfico

# --- 1. CARGA DE DATOS ---
# Lee el archivo CSV de calidad de agua y lo guarda en la variable 'df'
df = pd.read_csv("Documentos/CalidadAgua.csv")

# --- 2. PREPROCESAMIENTO (CODIFICACIÓN) ---
# Creamos un objeto traductor (LabelEncoder) para cada columna categórica (que contenga texto).
le_Fuente = LabelEncoder()
le_Temporada = LabelEncoder()
le_Apta = LabelEncoder()

# fit_transform lee las palabras únicas de cada columna y las reemplaza por números.
# Ejemplo Fuente: "Red" = 0, "Pozo" = 1, "Río" = 2.
# Esto es OBLIGATORIO porque el modelo matemático fallará si le pasas texto.
df["Fuente"] = le_Fuente.fit_transform(df["Fuente"])
df["Temporada"] = le_Temporada.fit_transform(df["Temporada"])
df["Apta"] = le_Apta.fit_transform(df["Apta"])

# --- 3. DEFINICIÓN DE VARIABLES ---
# X (Variables predictoras/Entradas): Quitamos la columna objetivo ("Apta").
X = df.drop("Apta", axis=1)

# y (Variable objetivo/Salida): Lo que queremos predecir (Si el agua es bebible/apta o no).
y = df["Apta"]

# --- 4. DIVISIÓN DE DATOS ---
# Separamos el 30% de los datos para el examen final del modelo (test) 
# y el 70% restante para que estudie y encuentre patrones (train).
# random_state=40 garantiza que los datos se revuelvan exactamente igual cada vez.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# --- 5. ENTRENAMIENTO DEL MODELO ---
# Creamos el árbol limitando su profundidad a 3 niveles (max_depth=3).
# PREGUNTA DE EXAMEN: ¿Por qué limitar la profundidad? Para evitar el "sobreajuste" (overfitting),
# es decir, evitar que el árbol crezca tanto que se aprenda los datos de memoria en lugar de generalizar.
modelo = DecisionTreeClassifier(max_depth=3)

# El modelo busca patrones en los datos de entrenamiento para aprender a decidir.
modelo.fit(X_train, y_train)

# --- 6. EVALUACIÓN ---
# Le pedimos al modelo que adivine la calidad del agua de los datos que guardamos para el examen (X_test).
y_pred = modelo.predict(X_test)

# classification_report imprime una tabla con:
# - Precision: De lo que predijo como "Apto", ¿cuánto era realmente "Apto"?
# - Recall: De todo el agua que era "Apta", ¿cuánta logró encontrar?
# - F1-score: El balance entre los dos anteriores.
print(classification_report(y_test, y_pred))

# --- 7. VISUALIZACIÓN ---
# Creamos un lienzo de 12 pulgadas de ancho por 6 de alto.
plt.figure(figsize=(12,6))

# Dibujamos el árbol visualmente. 
plot_tree(
    modelo, 
    feature_names=X.columns, # Muestra el nombre real de las columnas en cada división (ej. "Fuente <= 0.5")
    class_names=le_Apta.classes_.astype(str), # Muestra el resultado final en texto (ej. "Apta", "No Apta") en lugar de 0 o 1
    filled=True, # Colorea las cajitas según la clase que predomine
    rounded=True # Hace los bordes de las cajas redondos para que se vea más limpio
)

# Mostramos la ventana final con el resultado gráfico
plt.show()