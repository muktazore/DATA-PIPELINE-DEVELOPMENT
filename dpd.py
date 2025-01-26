import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Step 1: Load the data
# Example: Load a CSV file
input_file = "data.csv"  # Replace with your file path
data = pd.read_csv(input_file)

# Display the first few rows of the dataset
print("Initial Dataset:")
print(data.head())

# Step 2: Define preprocessing steps
# Identify numerical and categorical columns
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Define transformations for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define transformations for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 3: Split the data into training and testing sets
# Assuming the target column is named 'target'
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the preprocessing pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Step 5: Apply the pipeline to the training and testing data
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Display transformed data
print("\nTransformed Training Data:")
print(X_train_transformed)
print("\nTransformed Testing Data:")
print(X_test_transformed)

# Optional: Save the preprocessed data to files
pd.DataFrame(X_train_transformed).to_csv("X_train_preprocessed.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("X_test_preprocessed.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("\nData preprocessing, transformation, and loading completed successfully!")
