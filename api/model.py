from typing import List

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    instances: List[int]