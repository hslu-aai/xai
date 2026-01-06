import os

import numpy as np
from graphviz import Digraph
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor


def weight_to_color(weight: float, abs_max: float) -> str:
    w_clipped = max(-abs_max, min(weight, abs_max)) / abs_max
    if w_clipped >= 0:
        r, g, b = (int((1 - w_clipped) * 255), int((1 - w_clipped) * 255), 255)
    else:
        r, g, b = (255, int((1 + w_clipped) * 255), int((1 + w_clipped) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

mlp = MLPRegressor(
    hidden_layer_sizes=(8, 8, 8), activation="relu", max_iter=1000, random_state=42
)
mlp.fit(X, y)

dot = Digraph(format="svg")
dot.attr(rankdir="LR", nodesep="0.5", ranksep="4.0")

layer_sizes = [len(feature_names)] + list(mlp.hidden_layer_sizes) + [1]

node_id = lambda layer, index: f"L{layer}_N{index}"

for layer, size in enumerate(layer_sizes):
    with dot.subgraph() as s:
        s.attr(rank="same")  # Align nodes in the same layer horizontally
        for i in range(size):
            nid = node_id(layer, i)
            label = ""
            if layer == 0:
                label = feature_names[i]
            elif layer == len(layer_sizes) - 1:
                label = "Output"
            width = "1.0" if label == "" else "1.2"
            s.node(nid, label=label, shape="circle", width=width, fixedsize="true")

for layer in range(len(layer_sizes) - 1):
    weights = mlp.coefs_[layer]
    abs_max = max(abs(w) for node_weight in weights for w in node_weight)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            src = node_id(layer, i)
            dst = node_id(layer + 1, j)
            weight = weights[i, j]
            color = weight_to_color(weight, abs_max)
            dot.edge(src, dst, color=color)

output_path = os.path.join(os.pardir, "figures", "lecture1_nn")
dot.render(output_path, cleanup=True)
