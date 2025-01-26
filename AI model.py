import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Loads the dataset
data = pd.read_csv("obesity_data.csv")

# Display basic dataset information
print(data.head())
print(data.columns)
print(data.info())

# Clean column names This was done so no error occurs when writing target column
data.columns = data.columns.str.strip()

target_column = "ObesityCategory"
X = data.drop(target_column, axis=1)
y = data[target_column]

# Encoding the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# numerical and categorical columns
numerical_columns = ['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']
categorical_columns = ['Gender']

# One-hot encoding categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_categorical = encoder.fit_transform(X[categorical_columns])
categorical_feature_names = encoder.get_feature_names_out(categorical_columns)

# Combine numerical and categorical features
X_numerical = X[numerical_columns].to_numpy()
X_processed = np.hstack([X_numerical, X_categorical])

# Generate full feature names previus code had difference in features called and was giving error so this was included to make sure
all_feature_names = numerical_columns + list(categorical_feature_names)
print("Updated feature names:", all_feature_names)
print("Number of updated feature names:", len(all_feature_names))


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate Decision Tree model Ai was used t create logic for models
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Results:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_dt, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_dt, average='weighted'))
try:
    print("ROC-AUC:", roc_auc_score(y_test, dt_model.predict_proba(X_test), multi_class='ovr'))
except AttributeError:
    print("ROC-AUC not available for Decision Tree without predict_proba.")

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate Logistic Regression model
y_pred_lr = lr_model.predict(X_test)
print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_lr, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_lr, average='weighted'))
print("ROC-AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test), multi_class='ovr'))


plt.figure(figsize=(10, 6))

# Decision Tree ROC-AUC
try:
    dt_probs = dt_model.predict_proba(X_test)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs[:, 1], pos_label=1)
    roc_auc_dt = auc(fpr_dt, tpr_dt)
    plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {roc_auc_dt:.2f})")
except AttributeError:
    print("Decision Tree does not support predict_proba for ROC-AUC.")

# Logistic Regression ROC-AUC
lr_probs = lr_model.predict_proba(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs[:, 1], pos_label=1)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")

# Plot formatting
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Comparison")
plt.legend(loc="lower right")
plt.show()


feature_indices = dt_model.tree_.feature
valid_features = feature_indices[feature_indices != -2]
print("Valid feature indices (used for splits):", valid_features)
print("Feature names used in splits:", [all_feature_names[i] for i in valid_features])

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    dt_model,
    feature_names=all_feature_names,  # Correct feature names
    class_names=label_encoder.classes_,  # Class names
    filled=True
)
plt.show()
# data set used can be found here https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/code