# En este primer avance se define la estructura del proyecto
# siguiendo la metodología CRISP-DM, enfocada en IA.
# Aún no se implementa evaluación ni uso del modelo.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carga de datos

def cargar_datos(ruta):
    """
    Fase CRISP-DM: Comprensión de los datos
    """
    data = pd.read_csv(ruta)
    print("Datos cargados:")
    print(data.head())
    return data

# Preparación de datos
def preparar_datos(data):
    """
    Fase CRISP-DM: Preparación de datos
    """
    X = data[['horas_estudio']]
    y = data['calificacion']

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Modelado
def entrenar_modelo(X_train, y_train):
    """
    Fase CRISP-DM: Modelado
    
    Compatible con metodologías ágiles:
    - Scrum: se puede desarrollar en un sprint
    - Kanban: representa una etapa del flujo
    """
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

# Flujo principal del programa
def main():
    # Cargar datos
    data = cargar_datos("datos.csv")

    # Preparar datos
    X_train, X_test, y_train, y_test = preparar_datos(data)

    # Entrenar modelo
    modelo = entrenar_modelo(X_train, y_train)

    print("Modelo entrenado correctamente (fase inicial completada)")

if __name__ == "__main__":
    main()