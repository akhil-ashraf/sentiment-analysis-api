from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import os

# Load the trained model
model_path = "models/sentiment_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model not found! Please train the model first.")
    model = None

# Create FastAPI app
app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="Analyze sentiment of movie reviews",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Define request/response models
class ReviewRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.get("/")
def read_root():
    return FileResponse('src/static/index.html')

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Make prediction
    prediction = model.predict([request.text])[0]
    
    # Get prediction probability for confidence
    prediction_proba = model.predict_proba([request.text])[0]
    confidence = max(prediction_proba)
    
    # Convert to readable format
    sentiment = "positive" if prediction == 1 else "negative"
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=round(confidence, 3)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)