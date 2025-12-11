from data_load import load_data
from preprocess import split_data, create_preprocess_pipeline
from base_tree import get_base_model
from bagging_model import get_bagging_model
from boosting_model import get_boosting_model
from evaluate import evaluate_model
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_cv(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

def main():
    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess
    preprocess = create_preprocess_pipeline()
    X_train_prep = preprocess.fit_transform(X_train)
    X_test_prep = preprocess.transform(X_test)

    # BASE MODEL
    base = get_base_model()
    base.fit(X_train_prep, y_train)
    evaluate_model(base, X_test_prep, y_test, "Base Tree")
    run_cv(base, X_train_prep, y_train)

    # BAGGING MODEL
    bag = get_bagging_model()
    bag.fit(X_train_prep, y_train)
    evaluate_model(bag, X_test_prep, y_test, "Bagging")
    run_cv(bag, X_train_prep, y_train)

    # BOOSTING MODEL
    boost = get_boosting_model()
    boost.fit(X_train_prep, y_train)
    evaluate_model(boost, X_test_prep, y_test, "Boosting")
    run_cv(boost, X_train_prep, y_train)

if __name__ == "__main__":
    main()
