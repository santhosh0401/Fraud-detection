# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
# Assuming you have a CSV file with transaction data, replace 'your_dataset.csv' with your actual dataset
df = pd.read_csv('your_dataset.csv')

# Explore the data
print(df.head())

# Preprocess and normalize the data
# Drop unnecessary columns, handle missing values, etc.
# Example:
# df = df.drop(['unnecessary_column'], axis=1)
# df = df.dropna()

# Split the data into features (X) and target variable (y)
X = df.drop('fraud_label', axis=1)  # Assuming 'fraud_label' is the column indicating fraud or not
y = df['fraud_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using oversampling or undersampling
# Example using RandomOverSampler for oversampling
over_sampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)

# Example using RandomUnderSampler for undersampling
# under_sampler = RandomUnderSampler(random_state=42)
# X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Build the machine learning pipeline with a classification algorithm
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
