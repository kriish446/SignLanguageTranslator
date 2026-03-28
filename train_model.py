"""
Train RandomForest on collected landmark .npy files and save model.pkl.
"""

import glob
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def filename_to_label(basename: str) -> str:
    stem = os.path.splitext(basename)[0]
    return stem.replace("_", " ")


def main():
    paths = sorted(glob.glob(os.path.join("dataset", "*.npy")))
    if not paths:
        print("No .npy files in dataset/. Run collect_data.py first.")
        return

    X_list = []
    y_list = []
    for path in paths:
        arr = np.load(path)
        label = filename_to_label(os.path.basename(path))
        n = arr.shape[0]
        X_list.append(arr)
        y_list.extend([label] * n)

    X = np.vstack(X_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test) * 100
    y_pred = clf.predict(X_test)

    print(f"Test accuracy (%): {acc:.2f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Model saved! Run inference.py to start detecting.")


if __name__ == "__main__":
    main()
