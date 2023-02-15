import logging
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from src.features.build_features import custom_encoder, FeatureEngineer

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    COUNTER = 1
    PATH = str(sys.argv[COUNTER]) if len(
        sys.argv) > COUNTER else 'data/raw/train.txt'
    COUNTER += 1
    MAX_FEATURES = str(sys.argv[COUNTER]) if len(
        sys.argv) > COUNTER else 'sqrt'
    COUNTER += 1
    N_ESTIMATORS = int(sys.argv[COUNTER]) if len(sys.argv) > COUNTER else 50
    COUNTER += 1
    MAX_DEPTH = sys.argv[COUNTER] if len(sys.argv) > COUNTER else None
    if MAX_DEPTH == 'None':
        MAX_DEPTH = None
    COUNTER += 1
    MIN_SAMPLES_SPLIT = int(sys.argv[COUNTER]) if len(
        sys.argv) > COUNTER else 5
    COUNTER += 1
    MIN_SAMPLES_LEAF = int(sys.argv[COUNTER]) if len(sys.argv) > COUNTER else 1
    COUNTER += 1
    BOOTSTRAP = bool(sys.argv[COUNTER]) if len(sys.argv) > COUNTER else True

    data = pd.read_csv(PATH, delimiter=';',
                       names=['text', 'label'])

    texts = data['text']
    labels = data['label']

    labels = custom_encoder(labels)

    feature_engineer = FeatureEngineer()

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        bootstrap=BOOTSTRAP
    )

    pipeline = Pipeline(
        [('feature_engineer', feature_engineer), ('classifier', classifier)])

    text_train, text_test, label_train, label_test = train_test_split(
        texts, labels, test_size=0.2)

    with mlflow.start_run():

        pipeline.fit(text_train, label_train)

        label_pred = pipeline.predict(text_test)

        accuracy = accuracy_score(label_test, label_pred)
        preicsion = precision_score(label_test, label_pred)
        recall = recall_score(label_test, label_pred)

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("bootstrap", BOOTSTRAP)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("preicsion", preicsion)
        mlflow.log_metric("recall", recall)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                pipeline, "pipeline",
                registered_model_name="SentimentAnalysisPipeline")
        else:
            mlflow.sklearn.log_model(pipeline, "pipeline")

    mlflow.end_run()
