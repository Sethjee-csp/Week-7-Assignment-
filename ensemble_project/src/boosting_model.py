from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
def get_boosting_model():
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=80, learning_rate=0.8, random_state=42)
