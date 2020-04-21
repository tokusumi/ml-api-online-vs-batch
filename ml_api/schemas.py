from pydantic import BaseModel
from typing import List


class Text(BaseModel):
    text: str


class Data(BaseModel):
    data: List[Text]


class Output(Text):
    sentiment: str


class Pred(BaseModel):
    prediction: List[Output]
