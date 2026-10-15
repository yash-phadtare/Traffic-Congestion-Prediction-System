import pandas as pd

from traffic_system import ml


def test_regression_tuning_uses_ordered_window_and_refits_all_rows(monkeypatch):
    search_fit_indexes: list[list[int]] = []

    class RecordingPipeline:
        def __init__(self):
            self.fit_indexes: list[int] = []

        def fit(self, x_data, _y_data):
            self.fit_indexes = x_data.index.tolist()
            return self

    class FakeSearch:
        def __init__(self, **_kwargs):
            self.best_estimator_ = RecordingPipeline()

        def fit(self, x_data, _y_data):
            search_fit_indexes.append(x_data.index.tolist())
            return self

    monkeypatch.setattr(ml, "RandomizedSearchCV", FakeSearch)

    rows = 9_000
    x_train = pd.DataFrame({"hour": [index % 24 for index in range(rows)]})
    y_train = pd.Series(range(rows))

    fitted = ml._tune_pipeline(
        RecordingPipeline(),
        "RandomForestRegressor",
        x_train,
        y_train,
        "neg_root_mean_squared_error",
    )

    assert search_fit_indexes == [list(range(1_000, 9_000))]
    assert fitted.fit_indexes == list(range(9_000))
