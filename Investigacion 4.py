# La metodología principal utilizada en este proyecto es CRISP-DM,
# ya que está enfocada en proyectos de Inteligencia Artificial y análisis de datos.
# Además, se complementa con metodologías ágiles como Scrum/Kanban
# mediante la modularidad del código y la división en funciones.


# (Fase CRISP-DM: Comprensión técnica del entorno)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar datos
def cargar_datos(ruta):
    """
    Fase CRISP-DM: Comprensión de los datos
    
    Aquí se obtiene la información necesaria para resolver el problema.
    En IA, esta fase es clave porque define la calidad del modelo.
    """
    data = pd.read_csv(ruta)
    
    print("Primeros datos del dataset:")
    print(data.head())
    
    return data

# Preparar datos
def preparar_datos(data):
    """
    Fase CRISP-DM: Preparación de los datos
    
    Se seleccionan variables, se limpian datos y se dividen
    en entrenamiento y prueba.
    
    Esta fase demuestra la naturaleza iterativa de CRISP-DM,
    ya que puede repetirse varias veces para mejorar resultados.
    """
    
    # Variable independiente (input)
    X = data[['horas_estudio']]
    
    # Variable dependiente (output)
    y = data['calificacion']
    
    # División de datos (entrenamiento/prueba)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Modelado
def entrenar_modelo(X_train, y_train):
    """
    Fase CRISP-DM: Modelado
    
    Se entrena un modelo de Machine Learning.
    
    COMPATIBILIDAD CON OTRAS METODOLOGÍAS:
    - Scrum: este bloque puede desarrollarse en un sprint
    - XP: función reutilizable y clara
    - Kanban: etapa 'en progreso' dentro del flujo de trabajo
    """
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    return modelo

# Evaluar modelo
def evaluar_modelo(modelo, X_test, y_test):
    """
    Fase CRISP-DM: Evaluación
    
    Se mide el rendimiento del modelo para validar si cumple
    con el objetivo del proyecto.
    
    Esta fase permite regresar a etapas anteriores (iteración),
    lo cual es fundamental en proyectos de IA.
    """
    
    predicciones = modelo.predict(X_test)
    error = mean_squared_error(y_test, predicciones)
    
    print("Error del modelo:", error)
    
    return error

# Usar modelo para predicciones
def predecir(modelo, horas):
    """
    Fase CRISP-DM: Despliegue
    
    Se utiliza el modelo para hacer predicciones reales.
    
    COMPATIBILIDAD:
    - DevOps / MLOps: esta función representa el uso del modelo en producción
    """
    
    resultado = modelo.predict([[horas]])
    
    print(f"Predicción para {horas} horas de estudio:", resultado)
    
    return resultado

# Flujo principal del programa
def main():
    """
    Este flujo completo representa la aplicación de CRISP-DM
    en un proyecto de IA en Python dentro de Visual Studio.
    
    También refleja metodologías ágiles:
    - Código modular (Scrum / XP)
    - Flujo claro (Kanban)
    """
    
    # 1. Cargar datos
    data = cargar_datos("datos.csv")
    
    # 2. Preparar datos
    X_train, X_test, y_train, y_test = preparar_datos(data)
    
    # 3. Entrenar modelo
    modelo = entrenar_modelo(X_train, y_train)
    
    # 4. Evaluar modelo
    evaluar_modelo(modelo, X_test, y_test)
    
    # 5. Usar modelo
    predecir(modelo, 5)


if __name__ == "__main__":
    main()