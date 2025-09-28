import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def create_sample_data():
    """Create sample movie review data for training"""
    reviews = [
        "This movie was absolutely amazing! Great plot and acting.",
        "Terrible movie, waste of time and money.",
        "I loved every minute of it. Brilliant cinematography!",
        "Boring and predictable. Not recommended.",
        "Outstanding performance by all actors. Must watch!",
        "Very disappointing. Poor storyline.",
        "Fantastic movie with great special effects!",
        "Not worth watching. Very poor quality.",
        "Excellent direction and screenplay. Loved it!",
        "Awful movie. Complete disaster.",
        "Great entertainment! Highly recommended.",
        "Very bad acting and direction.",
        "Amazing story with perfect execution!",
        "Worst movie I've ever seen.",
        "Beautiful cinematography and music!",
        "Completely boring and uninteresting.",
        "Incredible performances by the entire cast.",
        "Poorly written script and bad direction.",
        "One of the best movies I've watched this year!",
        "Absolute waste of time. Terrible in every way."
    ]
    
    # 1 = positive, 0 = negative
    sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

def train_model():
    print("🎬 Training Sentiment Analysis Model...")
    
    # Create sample data
    df = create_sample_data()
    print(f"📊 Created dataset with {len(df)} reviews")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.3, random_state=42
    )
    
    print(f"🔄 Training on {len(X_train)} reviews, testing on {len(X_test)} reviews")
    
    # Create pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train model
    print("🤖 Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained! Accuracy: {accuracy:.2f}")
    
    # Test with some examples
    print("\n🧪 Testing with sample reviews:")
    test_reviews = [
        "This movie is fantastic!",
        "Boring and terrible film.",
        "Amazing cinematography and great acting!"
    ]
    
    predictions = model.predict(test_reviews)
    for review, pred in zip(test_reviews, predictions):
        sentiment = "😊 Positive" if pred == 1 else "😞 Negative"
        print(f"  '{review}' → {sentiment}")
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sentiment_model.pkl')
    print("✅ Model saved as models/sentiment_model.pkl")
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("\n🎉 Sentiment Analysis Model Ready!")