import logging
import pandas as pd

from src.models.predict_model import predict
from src.models.train_model import train_model


def main():
    '''
    Main function.
    '''
    data = pd.read_csv("data/raw/train.txt", delimiter=';',
                       names=['text', 'label'])

    model, feature_engineering = train_model(data)

    texts = ['I am so happy', 'I am so sad']
    predictions = predict(texts, model, feature_engineering)
    logging.info('Predictions are : %s', predictions)


if __name__ == "__main__":
    main()
