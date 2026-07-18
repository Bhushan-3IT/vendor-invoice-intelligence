from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

def train_random_forest(X_train,y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=1)

    scorer = make_scorer(f1_score)

    param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini"]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy=accuracy_score(y_test,preds)
    report=classification_report(y_test,preds)
    print(f"accurancy  : {accuracy:.2f}")
    print(report)






























# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "id": "d2d66da2-5642-4ee1-af11-00a1a3b5df96",
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "",
#    "name": ""
#   },
#   "language_info": {
#    "name": ""
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }
