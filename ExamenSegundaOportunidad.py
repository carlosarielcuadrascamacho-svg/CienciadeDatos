from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report
import pandas as pd 
import numpy as np

# Cargar el dataset
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

# Codificar variables categóricas
le_producto = LabelEncoder()
le_ciudad = LabelEncoder()
le_campaña = LabelEncoder()

df['Producto'] = le_producto.fit_transform(df['Producto'])
df['Ciudad'] = le_ciudad.fit_transform(df['Ciudad'])
df['Campaña_Marketing'] = le_campaña.fit_transform(df['Campaña_Marketing'])

df.loc[df['Ventas'] < 1000, 'Ventas'] = 0
df.loc[df['Ventas'] >= 1000, 'Ventas'] = 1

df = df.dropna()

print(df.to_string())

# Dividir el dataset en X y
X = df.drop('Ventas', axis=1) #Es todo el dataframe menos ventas
y = df['Ventas'] #Es la columna ventas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(modelo, filled=True, feature_names=X.columns, class_names=['Baja','Alta'])
plt.show()
