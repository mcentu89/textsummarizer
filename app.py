from fastapi import FastAPI
import uvicorn
import sys
import os
import traceback
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline
from pathlib import Path

text:str = "What is Text Summarization? Text summarization is the process of creating a concise and coherent summary of a longer text document. The goal is to capture the main ideas and key information while omitting less important details. Text summarization can be performed using various techniques, including extractive methods (selecting important sentences or phrases from the original text) and abstractive methods (generating new sentences that convey the same meaning as the original text). It is commonly used in applications such as news aggregation, document summarization, and content recommendation systems."

app = FastAPI()

@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training completed successfully.")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return Response(f"An error occurred during training: {str(e)}\n\n{tb}", status_code=500)
    
@app.post("/predict")
async def predict(text: str):
    try:
        prediction_pipeline = PredictionPipeline()
        output = prediction_pipeline.predict(text)
        return output
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return Response(f"An error occurred during prediction: {str(e)}\n\n{tb}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)