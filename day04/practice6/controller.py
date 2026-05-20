from fastapi import APIRouter

router = APIRouter( prefix='/api')

from service import usedCarTradeService

@router.post("/retrain")
async def learning(training_data: list[dict]):
    return await usedCarTradeService.retrain(training_data)

@router.post('/predict')
async def getSalePrice(used_car_data: dict):
    return await usedCarTradeService.pricePrediction(used_car_data)