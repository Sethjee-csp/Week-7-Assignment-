from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
def get_bagging_model():
    base_tree = DecisionTreeClassifier(random_state=42)
    return BaggingClassifier(estimator=base_tree, n_estimators=50, bootstrap=True, oob_score=True, random_state=42)
