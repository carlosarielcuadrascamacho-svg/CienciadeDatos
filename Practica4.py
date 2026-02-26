# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este código es tu plantilla para crear un "Árbol de Decisión" para CLASIFICACIÓN.
# Úsalo en tu examen cuando:
# 1. Tengas que predecir una "Categoría" o "Palabra" (ej. el tipo de cultivo), 
#    no un número continuo (como el rendimiento o la temperatura).
# 2. Tu dataset tenga columnas con TEXTO ("Arcilloso", "Orgánico") y necesites 
#    transformarlas a números para que el modelo pueda procesarlas (LabelEncoder).
# 3. Te pidan visualizar las "reglas" que la computadora aprendió dibujando el árbol.
# =============================================================================

import pandas as pd # Importa pandas para manejar la tabla de datos
from sklearn.model_selection import train_test_split # Para dividir los datos en Entrenamiento (Train) y Prueba (Test)
from sklearn.preprocessing import LabelEncoder # CLAVE: Para convertir variables categóricas (texto) a números
from sklearn.tree import DecisionTreeClassifier # El algoritmo de clasificación: Árbol de Decisión
from sklearn.metrics import classification_report # Genera un reporte completo de qué tan bien clasificó el modelo
from sklearn.tree import plot_tree # Herramienta específica para dibujar el árbol de decisión
import matplotlib.pyplot as plt # Para configurar el lienzo y mostrar el gráfico final

# --- 1. CARGA DE DATOS ---
# Lee el archivo CSV y lo guarda en el DataFrame 'df'
df = pd.read_csv("Documentos/agricultura.csv")

# --- 2. PREPROCESAMIENTO (CODIFICACIÓN) - ¡Pregunta de examen! ---
# Las computadoras no entienden palabras como "Arcilloso" o "Maíz", solo entienden números.
# Creamos un "traductor" (LabelEncoder) para cada columna de texto.
le_suelo = LabelEncoder()
le_fert = LabelEncoder()
le_cultivo = LabelEncoder()

# fit_transform lee todas las palabras diferentes de la columna, les asigna un número (0, 1, 2...)
# y reemplaza el texto original con esos números en el mismo DataFrame.
df["Tipo_Suelo"] = le_suelo.fit_transform(df["Tipo_Suelo"])
df["Fertilizante"] = le_fert.fit_transform(df["Fertilizante"])
df["Cultivo"] = le_cultivo.fit_transform(df["Cultivo"])

# --- 3. DEFINICIÓN DE VARIABLES ---
# X (Características): Son todas las columnas EXCEPTO la que queremos predecir.
# Usamos .drop("Cultivo", axis=1) para tirar la columna objetivo y quedarnos con el resto.
X = df.drop("Cultivo", axis=1)

# y (Objetivo/Etiqueta): Es la columna exacta que queremos que el modelo adivine.
y = df["Cultivo"]

# --- 4. DIVISIÓN DE DATOS ---
# Separamos los datos: 70% para que el modelo entrene (aprenda) y 30% para el examen final (test).
# random_state=40 asegura que la partición sea siempre la misma cada vez que corras el código.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# --- 5. ENTRENAMIENTO DEL MODELO ---
# Creamos el modelo de Árbol de Decisión. 
# ¡OJO! max_depth=3 limita el árbol a solo 3 "pisos" o niveles de altura. 
# Esto se hace para evitar el "Overfitting" (que el modelo memorice en lugar de aprender).
modelo = DecisionTreeClassifier(max_depth=3)

# .fit() hace que el modelo busque los patrones y reglas lógicas en los datos de entrenamiento.
modelo.fit(X_train, y_train)

# --- 6. EVALUACIÓN ---
# Le damos al modelo los datos de prueba (X_test) para que adivine los cultivos.
y_pred = modelo.predict(X_test)

# classification_report compara lo que el modelo adivinó (y_pred) contra la realidad (y_test).
# Te imprimirá métricas vitales como 'Precision', 'Recall' y 'F1-score' para cada tipo de cultivo.
print(classification_report(y_test, y_pred))

# --- 7. VISUALIZACIÓN DEL ÁRBOL ---
# Creamos un lienzo ancho (12x6) porque los árboles tienden a extenderse mucho hacia los lados.
plt.figure(figsize=(12,6))

# plot_tree dibuja la estructura de decisiones.
plot_tree(
    modelo, 
    feature_names=X.columns, # Le pone el nombre real a las variables (ej. "Humedad <= 15.5")
    class_names=le_cultivo.classes_.astype(str), # Traduce de vuelta los números (0,1,2) al nombre del cultivo para mostrarlo en las cajas finales.
    filled=True, # Pinta los cuadros de colores. Cada color representa un tipo de cultivo diferente.
    rounded=True # Hace que las esquinas de los cuadros se vean redondeadas (estética).
)

# Renderiza y abre la ventana con tu árbol dibujado
plt.show()