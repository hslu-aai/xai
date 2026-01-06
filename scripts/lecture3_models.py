import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Fixed seed for reproducibility
RANDOM_SEED = 123

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Split into training and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

# Define and fit models
linreg = LinearRegression().fit(X_train, y_train)
tree = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED).fit(
    X_train, y_train
)
svr = SVR(kernel="rbf").fit(X_train, y_train)
mlp = MLPRegressor(
    hidden_layer_sizes=(50, 30),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=RANDOM_SEED,
).fit(X_train, y_train)

# Fixed, nontrivial assignment
joblib.dump(svr, "a.pkl")
joblib.dump(linreg, "b.pkl")
joblib.dump(mlp, "c.pkl")
joblib.dump(tree, "d.pkl")

# Save only evaluation data for the exercise
np.savez("eval_data.npz", X_eval=X_eval, y_eval=y_eval)

print("Saved models as a.pkl, b.pkl, c.pkl, d.pkl")
print("Saved evaluation data as eval_data.npz")
