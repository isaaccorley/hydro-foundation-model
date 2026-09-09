# Intentionally empty: eval configs reference `src.models.*` / `src.datasets.*` by dotted
# path (hydra instantiate), and eager imports here would pull lightning/mlflow into the
# pretraining entrypoint, which doesn't need them.
