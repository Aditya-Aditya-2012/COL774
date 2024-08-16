# Code for part c
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

target_column = train_df.columns[-1]

X_train = train_df.iloc[:, :-1]
y_train = train_df[target_column]
X_test = test_df

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to reduce dimensions to a maximum of 300 features
pca = PCA(n_components=min(300, X_train_scaled.shape[1]), random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Training the LassoCV model with a more refined range of alphas
lasso = LassoCV(alphas=np.logspace(-6, 0, 100), cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_pca, y_train)

y_pred = lasso.predict(X_test_pca)

np.savetxt(output_file, y_pred, fmt='%.6f')
