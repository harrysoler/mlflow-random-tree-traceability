import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Harrizon Soler
    """)
    return


@app.cell
def _():
    import marimo as mo

    import mlflow
    import mlflow.sklearn

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, cross_validate

    return (
        ParameterGrid,
        RandomForestClassifier,
        StratifiedKFold,
        accuracy_score,
        cross_validate,
        f1_score,
        load_iris,
        mlflow,
        mo,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classroom version
    """)
    return


@app.cell
def _(RandomForestClassifier, load_iris, mlflow, model, train_test_split):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("final_project")

    with mlflow.start_run(run_name="run_3"):
        n_estimators = 200
        max_depth = 10

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        _data = load_iris()

        _X_train, _X_test, _y_train, _y_test = train_test_split(
            _data.data,
            _data.target,
            test_size=0.2,
            random_state=42
        )

        _model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(_X_train, _y_train)

        accuracy = model.score(_X_test, _y_test)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            _model, 
            name="iris_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"accuracy: {accuracy}")
        print(f"run id: {mlflow.active_run().info.run_id}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Version with more precision
    """)
    return


@app.function
def extract_metrics_from(scores: dict) -> dict:
    train_f1 = float(scores["train_f1"].mean())
    cv_f1 = float(scores["test_f1"].mean())

    return {
        "train_f1_macro": train_f1,
        "cv_f1_macro": cv_f1,
        "cv_f1_std": float(scores["test_f1"].std()),
        "cv_accuracy": float(scores["test_accuracy"].mean()),
        "fit_time_mean": float(scores["fit_time"].mean()),
        "f1_gap": train_f1 - cv_f1,
    }


@app.cell
def _(ParameterGrid, StratifiedKFold, load_iris, mlflow, train_test_split):
    # for now the only limitation I see is that I have to delete mlflow.db and mlruns/ every time I run this script for avoiding garbage accumulation in MLFLow.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("final_project")

    CV_FOLDS = 5
    RANDOM_STATE = 42

    # I moved the datased load and data separation here for avoiding redundancy in every run
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=42
    )

    results=[]

    possible_parameters = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 10, None]
    }

    combinations = ParameterGrid(possible_parameters)
    return (
        CV_FOLDS,
        RANDOM_STATE,
        X_test,
        X_train,
        combinations,
        cv,
        results,
        y_test,
        y_train,
    )


@app.cell
def _(
    CV_FOLDS,
    RANDOM_STATE,
    RandomForestClassifier,
    X_train,
    combinations,
    cross_validate,
    cv,
    mlflow,
    results,
    y_train,
):
    for params in combinations:
        name = f"run_n{params["n_estimators"]}_depth{params["max_depth"]}"
    
        with mlflow.start_run(run_name=name) as run:
            _model = RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE
            )

            scores = cross_validate(
                _model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "f1": "f1_macro",
                    "accuracy": "accuracy"
                },
                return_train_score=True
            )

            metrics = extract_metrics_from(scores)

            mlflow.log_params({
                **params,
                "random_state": 42,
                "cv_folds": CV_FOLDS,
                "test_size": 0.2
            })

            mlflow.log_metrics(metrics)

            result = {
                "run_id": run.info.run_id,
                "params": params,
                "cv_f1_macro": metrics["cv_f1_macro"]
            }

            results.append(result)
            print(result)
    return


@app.cell
def _(
    RANDOM_STATE,
    RandomForestClassifier,
    X_test,
    X_train,
    accuracy_score,
    f1_score,
    mlflow,
    results,
    y_test,
    y_train,
):
    best = max(results, key=lambda result: result["cv_f1_macro"])

    with mlflow.start_run(run_id=best["run_id"]):
        model = RandomForestClassifier(
            **best["params"], 
            random_state=RANDOM_STATE
        )
    
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        f1_macro_score = f1_score(y_test, predictions, average="macro")

        mlflow.log_metrics({
            "test_f1_macro": f1_macro_score,
            "test_accuracy": accuracy_score(y_test, predictions)
        })

        mlflow.sklearn.log_model(model, "iris_model", serialization_format="cloudpickle")

        print("selected configuration")
        print(best["params"])
        print(f"f1 macro test: {f1_macro_score}")
    return (model,)


if __name__ == "__main__":
    app.run()
