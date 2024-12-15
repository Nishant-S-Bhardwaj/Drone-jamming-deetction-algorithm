import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
csv_file = r'N:\tata_elexi_jamming_detection_model\combined_dataset.csv'
data = pd.read_csv(csv_file)

# Features (RSSI, PDR) and labels (Scenarios)
X = data[['RSSI', 'PDR']].values
y = data['Scenario'].values  # Ensure the column 'Scenario' contains the three scenarios

# Preprocessing: Scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define class labels (Scenarios)
class_labels = [
    "Scenario 1 (No Attack)",
    "Scenario 2 (Reactive Jammer Attack)",
    "Scenario 3 (Constant Jammer Attack)"
]

# ---- MLPClassifier Implementation ----
print("\nTraining MLPClassifier...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# Evaluate the MLPClassifier
y_pred_mlp = mlp.predict(X_test)

print("\nMLP Classification Report:")
print(classification_report(
    y_test, 
    y_pred_mlp, 
    target_names=class_labels
))

# Confusion matrix for MLP
cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=np.unique(y_test))
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=class_labels)
plt.figure(figsize=(8, 6))
disp_mlp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - MLPClassifier")
plt.show()

# ---- DecisionTreeClassifier Implementation ----
print("\nTraining DecisionTreeClassifier...")

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Generate predictions
y_pred_tree = clf.predict(X_test)

# Classification report for Decision Tree
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

# # Plot the Decision Tree
# plt.figure(figsize=(15, 10))
# plot_tree(
#     clf,
#     feature_names=['RSSI', 'PDR'],
#     class_names=class_labels,
#     filled=True,
#     rounded=True
# )
# plt.title("Decision Tree Visualization")
# plt.show()

# Confusion Matrix for Decision Tree
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tree, display_labels=class_labels)
plt.title("Confusion Matrix - DecisionTreeClassifier")
plt.show()

# ---- Debugging Unique Labels ----
print("\nUnique labels in y_test:", np.unique(y_test))
print("Unique labels in y_pred_mlp:", np.unique(y_pred_mlp))
print("Unique labels in y_pred_tree:", np.unique(y_pred_tree))

# ---- Summary ----
print("Comparison Complete. Review the classification reports and confusion matrices above.")

# ---- Custom Testing ----
print("\n--- Custom Testing ---")
while True:
    try:
        # Input RSSI and PDR values from the user
        custom_rssi = float(input("Enter custom RSSI value: "))
        custom_pdr = float(input("Enter custom PDR value: "))

        # Scale the custom input using the same scaler used for training
        custom_input = scaler.transform([[custom_rssi, custom_pdr]])

        # Predict using both MLPClassifier and DecisionTreeClassifier
        mlp_prediction = mlp.predict(custom_input)[0]
        tree_prediction = clf.predict(custom_input)[0]

        # Print the predictions
        # print(f"\nPredicted Scenario by MLPClassifier: {class_labels[mlp_prediction]}")
        # print(f"Predicted Scenario by DecisionTreeClassifier: {class_labels[tree_prediction]}\n")

    except ValueError:
        print("Invalid input. Please enter numerical values.")
    
    # Ask if the user wants to test another custom value
    repeat = input("Do you want to test another value? (yes/no): ").strip().lower()
    if repeat != "yes":
        print("Exiting custom testing.")
        break
# ---- Final Accuracy ----
mlp_accuracy = mlp.score(X_test, y_test)
tree_accuracy = clf.score(X_test, y_test)

print(f"\nFinal Accuracy of MLPClassifier: {mlp_accuracy * 100:.2f}%")
print(f"Final Accuracy of DecisionTreeClassifier: {tree_accuracy * 100:.2f}%")


# Predict using MLPClassifier and DecisionTreeClassifier
mlp_prediction = mlp.predict(custom_input)[0]
tree_prediction = clf.predict(custom_input)[0]

# Adjust predictions to zero-based indexing
mlp_label = class_labels[mlp_prediction - 1]
tree_label = class_labels[tree_prediction - 1]

print(f"\nPredicted Scenario by MLPClassifier: {mlp_label}")
print(f"Predicted Scenario by DecisionTreeClassifier: {tree_label}\n")

from sklearn.preprocessing import LabelEncoder

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Transforms {1, 2, 3} into {0, 1, 2}
class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

# Use inverse transform to decode predictions later
mlp_prediction = mlp.predict(custom_input)[0]
tree_prediction = clf.predict(custom_input)[0]

# Decode predictions to original labels
# mlp_label = label_encoder.inverse_transform([mlp_prediction])[0]
tree_label = label_encoder.inverse_transform([tree_prediction])[0]

print(f"\nPredicted Scenario by MLPClassifier: {mlp_label}")
print(f"Predicted Scenario by DecisionTreeClassifier: {tree_label}\n")
