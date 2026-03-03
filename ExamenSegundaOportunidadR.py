import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
#Cargar el dataset
df = pd.read_csv('DatasetVentas.csv')

df = df.dropna()

df['Cantidad_Vendida'] = df['Cantidad_Vendida'].astype(str).str.replace(' unidades', '', regex=False).astype(float)
df['Producto'] = df['Producto'].str.strip().str.capitalize()
df['Ciudad'] = df['Ciudad'].str.strip().str.capitalize()
df['Campaña_Marketing'] = df['Campaña_Marketing'].str.strip().str.capitalize()
df['Precio_Unitario'] = df['Precio_Unitario'].astype(str).str.replace('$', '').astype(float)
df['Precio_Unitario'] = df['Precio_Unitario'].astype(str).str.replace(',', '.').astype(float)
df['Descuento_pct'] = df['Descuento_pct'].astype(str).str.replace('%', '', regex=False).astype(float)
df['Descuento_pct'] = df['Descuento_pct'].astype(str).str.replace(',', '.', regex=False).astype(float)
df['Ventas'] = df['Ventas'].astype(str).str.replace(',', '.', regex=False).astype(float)

le_producto = LabelEncoder()
le_ciudad = LabelEncoder()
le_campaña = LabelEncoder()

df['Producto'] = le_producto.fit_transform(df['Producto'])
df['Ciudad'] = le_ciudad.fit_transform(df['Ciudad'])
df['Campaña_Marketing'] = le_campaña.fit_transform(df['Campaña_Marketing'])


df = df.dropna()

print(df.to_string())

X = df.drop('Ventas', axis=1)
y = df['Ventas'] 

slr = LinearRegression() 
slr.fit(X,y)

y_pred = slr.predict(X) 


plt.figure(figsize=(12,6)) 
plt.scatter(y, y_pred, color='red', label='Predicciones')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color = "black", lw=2) 

plt.xlabel("Ventas reales")
plt.ylabel("Ventas estimadas")
plt.title("Ventas reales vs estimadas")
plt.show()