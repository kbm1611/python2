from fastapi import FastAPI
import uvicorn
import controller

app = FastAPI()

if __name__ == '__main__':
    uvicorn.run( 'app:app', host = '0.0.0.0', port = 8000, reload = True)

app.include_router( controller.router )