from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi import HTTPException
from ml_api import schemas, io
from ml_api.ml import MockMLAPI
app = FastAPI()

ml = MockMLAPI()
ml.load()


@app.post('/prediction/online', response_model=schemas.Pred)
async def online_prediction(data: schemas.Data):
    preds = ml.predict(data.data)
    return {"prediction": preds}


def batch_predict(filename: str):
    """batch predict method for background process"""
    ml = MockMLAPI()
    ml.load()
    data = io.load_inputs(filename)
    pred = ml.predict(data)
    io.save_outputs(pred, filename)
    print('finished prediction')


@app.get('/prediction/batch')
async def batch_prediction(filename: str, background_tasks: BackgroundTasks):
    if io.check_outputs(filename):
        raise HTTPException(status_code=404, detail="the result of prediction already exists")

    background_tasks.add_task(batch_predict, filename)
    return {}


@app.post('/upload')
async def upload(data: schemas.Data):
    filename = io.save_inputs(data)
    return {"filename": filename}


@app.get('/download', response_model=schemas.Pred)
async def download(filename: str):
    if not io.check_outputs(filename):
        raise HTTPException(status_code=404, detail="the result of prediction does not exist")
    preds = io.load_outputs(filename)
    return {"prediction": preds}
