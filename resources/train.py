# Importar librerías
import mlflow
import mlflow.sklearn
# Importar librerías de scikit-learn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Configurar MLflow para usar SQLite como backend, es decir almacenar los experimentos en un archivo local llamado mlflow.db
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Configurar el experimento en MLflow, es decir el nombre del experimento donde se guardarán los resultados
mlflow.set_experiment("Proyecto_Final_Grado")

# Iniciar una nueva ejecución de MLflow, donde se registrarán los parámetros, métricas y el modelo entrenado
with mlflow.start_run(run_name="Baseline"):
    n_estimators = 100  # Número de árboles en el bosque, es decir, cuántos clasificadores se entrenarán y luego se combinarán para hacer la predicción final
    max_depth = None    # Profundidad máxima de los árboles, es decir, cuántas divisiones se harán en cada árbol antes de detenerse

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Cargar el dataset
    data = load_iris()

    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )
    # Entrenar el modelo de Random Forest con los parámetros especificados
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)
    # Evaluar el modelo con los datos de prueba y calcular la precisión
    accuracy = model.score(X_test, y_test)
    # Registrar la métrica de precisión en MLflow y guardar el modelo entrenado
    mlflow.log_metric("accuracy", accuracy)
    # Las versiones recientes de MLflow guardan los modelos con skops (formato seguro) y exigen
    # declarar como confiable la estructura interna de los árboles del Random Forest.
    mlflow.sklearn.log_model(model, name="iris_model",
                             skops_trusted_types=["sklearn.tree._tree.Tree"])
    # Imprimir la precisión y el ID de la ejecución en la consola
    print(f"Modelo registrado con accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")