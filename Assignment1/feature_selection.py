import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge

# Load data
train_file = sys.argv[1]
created_file = sys.argv[2]
selected_file = sys.argv[3]

train_df = pd.read_csv(train_file)
target_column = train_df.columns[-1]

X_train = train_df.iloc[:, :-1]
y_train = train_df[target_column]

# Identify low cardinality features
low_cardinality_features = [col for col in X_train.columns if X_train[col].nunique() <= 5]

# One-Hot Encoding for low cardinality features
ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe_train = pd.DataFrame(ohe.fit_transform(X_train[low_cardinality_features]), columns=ohe.get_feature_names_out(low_cardinality_features))

X_train = X_train.drop(columns=low_cardinality_features).reset_index(drop=True)
X_train = pd.concat([X_train.reset_index(drop=True), ohe_train], axis=1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Add Polynomial Features (degree 2 for quadratic features)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)

# Feature names for polynomial features
poly_feature_names = poly.get_feature_names_out(X_train.columns)

# Compute weights for each sample to reduce the influence of outliers
errors = np.abs(y_train - np.median(y_train))
weights = np.where(errors > np.percentile(errors, 90), 0, 1.0)  # 0 weight for top 10% outliers

# Ridge Regression with the best lambda (alpha)
ridge = Ridge(alpha=5)
ridge.fit(X_train_poly, y_train, sample_weight=weights)

# Select the top 300 features based on the absolute value of the coefficients
top_300_indices = np.argsort(np.abs(ridge.coef_))[-300:]
selected_features = np.zeros(X_train_poly.shape[1], dtype=int)
selected_features[top_300_indices] = 1

# Write created.txt with all feature names
with open(created_file, 'w') as f:
    for feature in poly_feature_names:
        f.write(f"{feature}\n")

# Write selected.txt with 1 for top 300 features and 0 for the rest
with open(selected_file, 'w') as f:
    for selected in selected_features:
        f.write(f"{selected}\n")
