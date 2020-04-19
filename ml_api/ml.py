from random import choice
from time import sleep
from typing import List
from ml_api import schemas
from ml_api import io


def mock_ml_api(texts: List[schemas.Text]):
    """
    mock sentiment analysis
    might include followings,
    - Load data
    - Preprocess
    - Prediction
    - Post-process
    """
    sleep(10)
    preds = [choice(['happy', 'sad', 'angry']) for i in range(len(texts))]
    out = [{'text': t.text, 'sentiment': s} for t, s in zip(texts, preds)]
    return out


def mock_batch_ml_api(filename: str):
    data = io.load_inputs(filename)
    pred = mock_ml_api(data)
    io.save_outputs(pred, filename)
    print('finished prediction')
