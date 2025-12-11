from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def create_preprocess_pipeline():
    pipeline = Pipeline([("scaler", StandardScaler())])
    return pipeline
