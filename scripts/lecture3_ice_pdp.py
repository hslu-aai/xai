# !git clone https://github.com/stedy/Machine-Learning-with-R-datasets.git

import os

import pandas as pd

df = pd.read_csv(os.path.join("Machine-Learning-with-R-datasets", "insurance.csv"))

target = "charges"
features = ["age", "sex", "bmi", "children", "smoker", "region"]
X = df[features]
y = df[target]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

from sklearn.model_selection import train_test_split

X_train, X_validtest, y_train, y_validtest = train_test_split(
    X, y, test_size=0.5, random_state=1
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_validtest, y_validtest, test_size=0.5, random_state=2
)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline


model = GradientBoostingRegressor(random_state=42)
pipeline = make_pipeline(preprocessor, model)
pipeline.fit(X_train, y_train)

features_transformed = [s.split("__")[-1] for s in preprocessor.get_feature_names_out()]
X_train_transformed = preprocessor.transform(X_train)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=features_transformed)
X_valid_transformed = preprocessor.transform(X_valid)
X_valid_transformed = pd.DataFrame(X_valid_transformed, columns=features_transformed)

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

model = model.fit(X_train_transformed, y_train)

_, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 4), constrained_layout=True)
display = PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_valid_transformed,
    features=["age", "bmi", "children"],
    ax=ax,
    kind="both",
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
)
for axis in display.axes_:
    axis.set_ylabel("insurance claims [USD]")
plt.savefig(os.path.join(os.pardir, "figures", "lecture3_ice_pdp.svg"))

_, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 4), constrained_layout=True)
display = PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_valid_transformed,
    features=["age", "bmi", "children"],
    ax=ax,
    kind="both",
    centered=True,
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
)
for axis in display.axes_:
    axis.set_ylabel("insurance claims [USD]")
plt.savefig(os.path.join(os.pardir, "figures", "lecture3_ice_pdp_centered.svg"))
