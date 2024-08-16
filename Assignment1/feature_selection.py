import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

train_file = sys.argv[1]  # 'train.csv'
created_file = sys.argv[2]  # 'created.txt'
selected_file = sys.argv[3]  # 'selected.txt'

train_df = pd.read_csv(train_file)

X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(X)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(poly_features)

# Feature Selection using LassoCV
lasso = LassoCV(cv=5, max_iter=10000, random_state=42).fit(scaled_features, y)

selected_features = np.where(lasso.coef_ != 0)[0]
selected_feature_names = [poly.get_feature_names_out(input_features=X.columns)[i] for i in selected_features]

np.savetxt(created_file, scaled_features, delimiter=',')
with open(selected_file, 'w') as f:
    for feature in selected_feature_names:
        f.write(f"{feature}\n")
