from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi import HTTPException
from ml_api import schemas, io
from ml_api.ml import mock_ml_api, mock_batch_ml_api
app = FastAPI()


@app.post('/prediction/online')
async def online_prediction(data: schemas.Data):
    preds = mock_ml_api(data.data)
    return {"prediction": preds}


@app.get('/prediction/batch')
async def batch_prediction(filename: str, background_tasks: BackgroundTasks):
    if io.check_outputs(filename):
        raise HTTPException(status_code=404, detail="the result of prediction already exists")

    background_tasks.add_task(mock_batch_ml_api, filename)
    return {}


@app.post('/upload')
async def upload(data: schemas.Data):
    filename = io.save_inputs(data)
    return {"filename": filename}


@app.get('/download')
async def download(filename: str):
    preds = io.load_outputs(filename)
    return {"prediction": preds}
