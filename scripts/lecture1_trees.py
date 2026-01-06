import os

from graphviz import Source
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

model = RandomForestRegressor(
    n_estimators=4, max_depth=4, max_samples=128, random_state=42
)
model.fit(X, y)

for i, estimator in enumerate(model.estimators_):
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = Source(dot_data)
    graph.render(
        filename=os.path.join(os.pardir, "figures", f"lecture1_tree{i+1}"),
        format="svg",
        cleanup=True,
    )
