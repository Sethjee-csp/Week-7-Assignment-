from sklearn.tree import DecisionTreeClassifier
def get_base_model():
    return DecisionTreeClassifier(max_depth=None, random_state=42)
