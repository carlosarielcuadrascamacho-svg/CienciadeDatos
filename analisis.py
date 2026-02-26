import pandas as pd  # Importa pandas: la librería esencial para estructurar, manipular y analizar tablas de datos (DataFrames).
import matplotlib.pyplot as plt  # Importa pyplot de matplotlib: la herramienta principal para renderizar gráficas.

# --- Carga de Datos ---
# pd.read_csv() lee la ruta del archivo y transforma el texto en una tabla manejable (DataFrame) asignada a la variable 'df'.
df = pd.read_csv('Documentos/calificaciones.csv')

# Imprime la tabla completa en la terminal. Vital en el examen para asegurar que no hay errores de lectura en las columnas.
print(df)

# --- Análisis de Datos ---

# df['Calificacion'] aísla esa columna en específico. .mean() calcula la media aritmética de todos los registros ahí contenidos.
print('Promedio:', df['Calificacion'].mean())

# .max() recorre la columna seleccionada y te devuelve el valor más alto (la mejor calificación).
print('Máxima:', df['Calificacion'].max())

# .min() recorre la columna seleccionada y te devuelve el valor más bajo (la peor calificación).
print('Mínima:', df['Calificacion'].min())

# --- Gráfica ---

# plt.bar(eje_X, eje_Y) construye una gráfica de barras. El primer dato alimenta el eje horizontal y el segundo la altura de las barras.
plt.bar(df['Nombre'], df['Calificacion'])

# plt.title() define el título superior de tu lienzo para darle contexto a la visualización.
plt.title('Calificaciones de alumnos')

# plt.xlabel() y plt.ylabel() etiquetan los ejes para que el profesor (o cualquier persona) entienda qué representa cada línea.
plt.xlabel('Alumno')
plt.ylabel('Calificación')

# plt.show() ejecuta el renderizado final y abre una ventana en tu computadora para mostrar la gráfica dibujada.
plt.show()

# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este script es tu herramienta de "Análisis Exploratorio de Datos" (EDA) básico.
# Te sirve en el examen para:
# 1. Cargar rápidamente un dataset desde un archivo CSV.
# 2. Extraer métricas estadísticas fundamentales de una columna (promedio, máximo, mínimo).
# 3. Generar una visualización rápida (gráfica de barras) para comparar categorías de forma visual.
# Es el paso cero perfecto para entender tus datos antes de aplicar cualquier modelo.
# =============================================================================
