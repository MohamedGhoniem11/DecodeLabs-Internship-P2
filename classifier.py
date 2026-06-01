from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def main():
    print("=" * 55)
    print("  DecodeBot — Data Classification Using AI")
    print("  Project 2: Supervised Learning with Iris Dataset")
    print("=" * 55)

    # Step 1: Load and understand the dataset
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print(f"\nFeatures: {feature_names}")
    print(f"Target classes: {target_names}")
    print(f"Dataset shape: {X.shape}")
    print(f"Samples per class: {np.bincount(y)}")

    # Step 2: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Step 3: Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print(f"\nModel: KNeighborsClassifier(k=3)")
    print("Training complete.")

    # Step 4: Evaluate on test data
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.2f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Step 5: Classify a new custom input
    print("-" * 55)
    print("Classifying a new flower:")
    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = knn.predict(new_flower)
    print(f"  Measurements: {new_flower[0]}")
    print(f"  Predicted species: {target_names[prediction[0]]}")

    # Show prediction probabilities
    probs = knn.predict_proba(new_flower)
    for i, name in enumerate(target_names):
        print(f"  Confidence for {name}: {probs[0][i]:.2f}")

    print("=" * 55)
    print("  Project 2 complete!")


if __name__ == "__main__":
    main()
