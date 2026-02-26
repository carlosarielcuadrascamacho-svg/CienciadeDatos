# =============================================================================
# ¿PARA QUÉ ES BUENO ESTE CÓDIGO?
# Este código es clave cuando necesitas predecir un evento basándote en eventos anteriores
# (por ejemplo: usar datos del viernes y sábado para predecir el domingo).
# Te sirve en el examen para:
# 1. Filtrar un dataset para trabajar solo con una parte de los datos (limpieza de datos).
# 2. Aplicar Regresión Lineal Múltiple con 'statsmodels' (domingo depende de viernes y sábado).
# 3. Guardar la gráfica generada como un archivo de imagen (.png) para un reporte.
# =============================================================================

import pandas as pd # Para el manejo y estructuración de los datos
import statsmodels.formula.api as smf # Para crear el modelo estadístico detallado
import matplotlib.pyplot as plt # Para la visualización gráfica

# --- 1. CARGA DE DATOS ---
# Cargamos el archivo CSV de accidentes en el DataFrame 'df'
df = pd.read_csv("Documentos/sct_70_accidentes_dia.csv")

# --- 2. PREPARACIÓN Y FILTRADO (¡Muy importante en exámenes!) ---
# A veces no quieres usar todos los datos. Aquí le decimos a pandas:
# "Tráeme solo las filas donde la columna 'accidentes' diga exactamente 'accidentes'".
# El .copy() al final es una buena práctica para que Python cree una tabla nueva e independiente 
# y no te marque alertas raras cuando le agregues columnas más adelante.
df_filtrado = df[df['accidentes'] == 'accidentes'].copy()

# --- 3. CREACIÓN DEL MODELO ---
# Definimos nuestra Regresión Lineal.
# La fórmula "domingo ~ viernes + sabado" significa: 
# y (lo que quiero adivinar: accidentes del domingo) depende de X (viernes y sábado).
# Le pasamos nuestro 'df_filtrado', no el 'df' original.
modelo = smf.ols(formula="domingo ~ viernes + sabado", data=df_filtrado).fit()

# --- 4. REPORTE ESTADÍSTICO ---
# Imprime el análisis matemático complejo. 
# En el examen, si un coeficiente es positivo, significa que a más accidentes el viernes/sábado,
# más accidentes habrá el domingo.
print("--- RESUMEN ESTADÍSTICO DE ACCIDENTES ---")
print(modelo.summary())

# --- 5. PREDICCIONES ---
# El modelo usa los datos reales de viernes y sábado para adivinar cómo debió ser el domingo.
# Guardamos este resultado matemático en la nueva columna 'Prediccion_Domingo'.
df_filtrado["Prediccion_Domingo"] = modelo.predict(df_filtrado)

# --- 6. VISUALIZACIÓN ---
plt.figure(figsize=(8, 6)) # Crea el lienzo de 8x6 pulgadas

# plt.scatter() dibuja la comparativa: Eje X (Domingo real) vs Eje Y (Domingo que adivinó el modelo).
# Los puntos naranjas representan cada registro (probablemente cada estado o semana).
plt.scatter(df_filtrado["domingo"], df_filtrado["Prediccion_Domingo"], color='orange', label='Estados (Datos)')

# Línea de referencia (Modelo Perfecto)
# Es nuestra "línea de acierto al 100%". Si un punto naranja cae exactamente aquí, 
# el modelo predijo el valor perfecto. 'linestyle=--' hace que la línea sea punteada.
plt.plot(
    [df_filtrado["domingo"].min(), df_filtrado["domingo"].max()],
    [df_filtrado["domingo"].min(), df_filtrado["domingo"].max()],
    color='blue', linestyle='--', label='Predicción Ideal'
)

# Etiquetas y Estética de la gráfica
plt.xlabel("Accidentes Reales (Domingo)")
plt.ylabel("Accidentes Estimados (Domingo)")
plt.title("Regresión Lineal: Predicción de Accidentes Dominicales")
plt.legend() # Muestra la guía de qué significa cada color
plt.grid(True, alpha=0.3) # Activa la cuadrícula, 'alpha' la hace un poco transparente para que no estorbe.

# --- GUARDAR Y MOSTRAR ---
# plt.savefig() toma una "foto" de tu gráfica y la guarda en tu computadora como una imagen.
# ¡Ojo! Esto debe ir SIEMPRE ANTES de plt.show(), si lo pones después, te guardará una imagen en blanco.
plt.savefig("grafica_accidentes.png")
print("\n-> Gráfica generada: grafica_accidentes.png")

# Muestra la ventana en pantalla
plt.show()