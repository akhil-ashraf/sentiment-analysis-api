import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_training():
    """Test that model training works"""
    from src.train import train_model
    model = train_model()
    assert model is not None

def test_prediction():
    """Test basic prediction functionality"""
    import joblib
    import os
    
    # Check if model exists
    model_path = "models/sentiment_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # Test prediction
        prediction = model.predict(["This is a great movie!"])
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]