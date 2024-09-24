import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

train_file = sys.argv[1]  
created_file = sys.argv[2]  
selected_file = sys.argv[3] 

train_df = pd.read_csv(train_file)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

# Drop the 'Birth Weight' column and add it to the beginning of the columns list
X_train = X_train.drop(columns=['Birth Weight'])
feature_names = ['Birth Weight']  # Start with the dropped column

# Identify numerical and categorical columns
numerical_cols = ['Total Costs', 'Length of Stay']
categorical_cols = X_train.columns.difference(numerical_cols)

# Define preprocessing for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)

# Get feature names after preprocessing
preprocessed_feature_names = preprocessor.get_feature_names_out()

# Combine 'Birth Weight' with the preprocessed feature names
feature_names.extend(preprocessed_feature_names)

# Apply ANOVA-based feature selection
feature_selector = SelectKBest(score_func=f_classif, k=900)
X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)

selected_indices = feature_selector.get_support(indices=True)

# Write all feature names to created_file
with open(created_file, 'w') as f:
    # Include 'Birth Weight' with 0 (not selected)
    f.write('Birth Weight\n')
    for feature in preprocessed_feature_names:
        f.write(f"{feature}\n")

# Write 1 or 0 for selected features to selected_file
selected_features = np.zeros(len(feature_names), dtype=int)
# 'Birth Weight' is not part of the selected features, so it's always 0
selected_features[1 + np.array(selected_indices)] = 1

with open(selected_file, 'w') as f:
    for selected in selected_features:
        f.write(f"{selected}\n")