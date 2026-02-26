# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este código sirve para crear un modelo de "Regresión Lineal Múltiple" usando 'statsmodels'.
# Úsalo en el examen cuando necesites:
# 1. Predecir un valor numérico continuo (variable dependiente 'y') basándote en 
#    DOS O MÁS variables (independientes 'X'). En este caso, predecir el rendimiento.
# 2. Obtener un reporte estadístico súper detallado (R-cuadrado, p-values, coeficientes)
#    para saber qué variables realmente importan y cuáles no.
# 3. Visualizar gráficamente qué tan exactas fueron las predicciones frente a la realidad.
# =============================================================================

import pandas as pd # Librería para cargar y manipular la tabla de datos (DataFrames)
import statsmodels.formula.api as smf # Librería para crear modelos estadísticos usando fórmulas matemáticas (estilo R)
import matplotlib.pyplot as plt # Librería para crear y personalizar las gráficas

# --- 1. CARGA DE DATOS ---
# Carga el archivo CSV en un DataFrame llamado 'df'. 
# Asegúrate de que el archivo esté en la misma carpeta o ajusta la ruta.
df = pd.read_csv("Documentos/agricultura.csv")

# --- 2. CREACIÓN Y ENTRENAMIENTO DEL MODELO ---
# 'smf.ols' crea un modelo de Mínimos Cuadrados Ordinarios (Ordinary Least Squares).
# La 'formula' funciona así: "Lo_que_quiero_predecir ~ Variable_1 + Variable_2 + Variable_3"
# La virgulilla (~) separa la 'y' (Rendimiento) de las 'X' (Riego, Temperatura, Humedad).
modelo = smf.ols(
    formula="Rendimiento_t_ha ~ Riego_mm + Temperatura_C + Humedad", 
    data=df
).fit() # .fit() es el comando clave que ejecuta los cálculos matemáticos para que el modelo "aprenda" de los datos.

# --- 3. REPORTE ESTADÍSTICO ---
# Imprime una tabla gigante en la terminal. 
# En el examen fíjate en el "R-squared" (qué tan bueno es el modelo) y en 
# "P>|t|" (si es menor a 0.05, esa variable influye significativamente).
print(modelo.summary())

# --- 4. PREDICCIONES ---
# modelo.predict(df) toma las columnas X (Riego, Temp, Humedad) y aplica la fórmula aprendida.
# El resultado son las "adivinanzas" matemáticas del modelo, las cuales guardamos 
# en una nueva columna de la tabla llamada 'Estimacion'.
df["Estimacion"] = modelo.predict(df)

# --- 5. VISUALIZACIÓN ---
# plt.figure(figsize=(8, 6)) crea el "lienzo" en blanco y le da un tamaño de 8x6 pulgadas.
plt.figure(figsize=(8, 6))

# plt.scatter() dibuja puntos sueltos. Eje X: Datos reales, Eje Y: Datos que predijo el modelo.
# Si el modelo fuera perfecto, los puntos formarían una línea recta diagonal exacta.
plt.scatter(df["Rendimiento_t_ha"], df["Estimacion"], color='blue', label='Datos reales')

# plt.plot() dibuja la "línea de perfección". Va desde el valor mínimo real hasta el máximo real.
# Es nuestra regla de medición: entre más cerca estén los puntos azules a esta línea roja, mejor es el modelo.
plt.plot(
    [df["Rendimiento_t_ha"].min(), df["Rendimiento_t_ha"].max()],
    [df["Rendimiento_t_ha"].min(), df["Rendimiento_t_ha"].max()],
    color='red', 
    linewidth=2,
    label='Modelo Perfecto'
)

# Configuraciones de estética para que la gráfica sea presentable en tu examen
plt.xlabel("Rendimiento Real (t/ha)")          # Etiqueta del eje horizontal
plt.ylabel("Rendimiento Estimado (t/ha)")      # Etiqueta del eje vertical
plt.title("Práctica 2: Regresión Lineal - Comparativa de Precisión") # Título principal
plt.legend() # Muestra el cuadrito que explica qué es el color azul y qué es el rojo (los 'label')
plt.grid(True) # Dibuja una cuadrícula de fondo para leer mejor las coordenadas de los puntos

# Renderiza la gráfica y la muestra en una ventana
plt.show()