import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar el dataset
file_path = r"C:\Users\s2dan\OneDrive\Documentos\WorkSpace\Proyect_AI\ObesityDataSet_raw_and_data_sinthetic.csv"
data = pd.read_csv(file_path)

# Codificar las variables categóricas
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Dividir el dataset en características (X) y la etiqueta (y)
X = data.drop('NObeyesdad', axis=1)  # Características
y = data['NObeyesdad']  # Etiqueta

# Inicializar el clasificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Lista para almacenar las precisiones
accuracies_80_20 = []
accuracies_50_50 = []

# Realizar 100 particiones con 80/20
for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_80_20.append(accuracy)

# Mediana de la precisión con 80/20
median_accuracy_80_20 = np.median(accuracies_80_20)

# Realizar 100 particiones con 50/50
for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_50_50.append(accuracy)

# Mediana de la precisión con 50/50
median_accuracy_50_50 = np.median(accuracies_50_50)

# Mostrar los resultados
print(f"Mediana de la precisión (80/20) después de 100 asignaciones: {median_accuracy_80_20 * 100:.2f}%")
print(f"Mediana de la precisión (50/50) después de 100 asignaciones: {median_accuracy_50_50 * 100:.2f}%")
