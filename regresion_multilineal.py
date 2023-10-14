import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos desde tu dataset (reemplaza 'tu_dataset.csv' con el nombre y ubicación real de tu archivo)
data = pd.read_csv('Student_Performance.csv')

# Dividir los datos en características (X) y la variable objetivo (y)
X = data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']

# Convertir la columna 'Extracurricular_activities' a valores binarios (1 si es 'Yes', 0 si es 'No')
X['Extracurricular Activities'] = (data['Extracurricular Activities'] == 'Yes').astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error cuadrático medio: {mse}')
print(f'Puntaje R-cuadrado (R2): {r2}')

# Crear un gráfico de dispersión para visualizar las predicciones vs. los valores reales
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs. Predicciones')
plt.show()