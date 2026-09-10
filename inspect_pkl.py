import pickle

# Load and inspect the Random Forest model
with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

print("=== RANDOM FOREST MODEL INFO ===")
print("Model Type:", type(rf_model))
print("Number of Decision Trees:", rf_model.n_estimators)
print("Features expected:", rf_model.n_features_in_)
print("Classes learned:", rf_model.classes_)

# Load and inspect the SVM model
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

print("\n=== SVM MODEL INFO ===")
print("Model Type:", type(svm_model))
print("Kernel Used:", svm_model.kernel)
print("Classes learned:", svm_model.classes_)