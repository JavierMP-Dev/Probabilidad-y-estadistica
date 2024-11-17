import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = defaultdict(lambda: defaultdict(float))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes, class_counts = np.unique(y, return_counts=True)
        
        # Calcular probabilidades a priori P(C_k)
        self.class_priors = {c: count / n_samples for c, count in zip(classes, class_counts)}

        # Calcular P(x_i | C_k) para cada característica y clase
        for c in classes:
            X_c = X[y == c]
            for i in range(n_features):
                feature_values, counts = np.unique(X_c[:, i], return_counts=True)
                for val, count in zip(feature_values, counts):
                    self.feature_probs[i][(val, c)] = count / X_c.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for c in self.class_priors.keys():
                score = np.log(self.class_priors[c])
                for i, val in enumerate(x):
                    score += np.log(self.feature_probs[i].get((val, c), 1e-6))  # Smoothing
                class_scores[c] = score
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

def cargar_datos(archivo):
    """
    Carga los datos desde un archivo .txt
    :param archivo: Nombre del archivo.
    :return: Características (X) y etiquetas (y).
    """
    X, y = [], []
    with open(archivo, "r") as f:
        for linea in f:
            partes = linea.strip().split(",")
            X.append(partes[:2])  # Primeras dos columnas son características
            y.append(partes[2])  # Última columna es la etiqueta
    return np.array(X), np.array(y)

# Cargar datos desde "datos.txt"
archivo_datos = "datos.txt"
X, y = cargar_datos(archivo_datos)

# Dividir en entrenamiento y prueba (80% - 20%)
n_entrenamiento = int(0.8 * len(X))
X_train, X_test = X[:n_entrenamiento], X[n_entrenamiento:]
y_train, y_test = y[:n_entrenamiento], y[n_entrenamiento:]

# Entrenar el modelo
modelo = NaiveBayesClassifier()
modelo.fit(X_train, y_train)

# Evaluar en el conjunto de prueba
predicciones = modelo.predict(X_test)
precision = np.mean(predicciones == y_test)

print(f"Precisión en el conjunto de prueba: {precision:.2f}")
