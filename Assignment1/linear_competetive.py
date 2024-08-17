# Root Mean Squared Error of the best 90% predictions: 9506.458067301632
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# Load data
train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

target_column = train_df.columns[-1]

X_train = train_df.iloc[:, :-1]
y_train = train_df[target_column]
X_test = test_df

# Identify low cardinality features
low_cardinality_features = [col for col in X_train.columns if X_train[col].nunique() <= 4]

# One-Hot Encoding for low cardinality features
ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe_train = pd.DataFrame(ohe.fit_transform(X_train[low_cardinality_features]), columns=ohe.get_feature_names_out(low_cardinality_features))
ohe_test = pd.DataFrame(ohe.transform(X_test[low_cardinality_features]), columns=ohe.get_feature_names_out(low_cardinality_features))

X_train = X_train.drop(columns=low_cardinality_features).reset_index(drop=True)
X_test = X_test.drop(columns=low_cardinality_features).reset_index(drop=True)

X_train = pd.concat([X_train.reset_index(drop=True), ohe_train], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), ohe_test], axis=1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add Polynomial Features (degree 2 for quadratic features)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Compute weights for each sample to reduce the influence of outliers
# Here we use the distance from the median as a simple method
errors = np.abs(y_train - np.median(y_train))
weights = np.where(errors > np.percentile(errors, 90), 0, 1.0)  # 0 weight for top 10% outliers

# Feature Selection or Linear Regression with regularization (using Ridge with the best lambda)
ridge = Ridge(alpha=5)  # The best lambda from previous Ridge regression analysis
ridge.fit(X_train_poly, y_train, sample_weight=weights)

# Select the features with non-zero coefficients
selected_features_mask = np.abs(ridge.coef_) > 1e-5
X_train_selected = X_train_poly[:, selected_features_mask]
X_test_selected = X_test_poly[:, selected_features_mask]

# Train the Linear Regression model on the selected features (with weighted samples)
linear_reg = LinearRegression()
linear_reg.fit(X_train_selected, y_train, sample_weight=weights)

y_pred = linear_reg.predict(X_test_selected)

np.savetxt(output_file, y_pred, fmt='%.6f')
