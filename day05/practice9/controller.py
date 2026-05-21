from fastapi import APIRouter

router = APIRouter(prefix='/api')

from service import fashionSystemService

# api/retrain
@router.post('/retrain')
async def learning(training_data : list[dict]):
    return await fashionSystemService.retrain( training_data )

# api/predict
@router.post('/predict')
async def getCategory(data: dict):
    return await fashionSystemService.categoryPredict( data )