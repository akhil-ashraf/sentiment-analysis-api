import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

def create_larger_dataset():
    """Create a much larger and more diverse movie review dataset"""
    positive_reviews = [
        "This movie was absolutely amazing! Great plot and acting.",
        "I loved every minute of it. Brilliant cinematography!",
        "Outstanding performance by all actors. Must watch!",
        "Fantastic movie with great special effects!",
        "Excellent direction and screenplay. Loved it!",
        "Great entertainment! Highly recommended.",
        "Amazing story with perfect execution!",
        "Beautiful cinematography and music!",
        "Incredible performances by the entire cast.",
        "One of the best movies I've watched this year!",
        "Absolutely brilliant! Masterful storytelling and direction.",
        "Stunning visuals and compelling narrative. A true masterpiece!",
        "Phenomenal acting and incredible attention to detail.",
        "This film exceeded all my expectations. Simply wonderful!",
        "Captivating from start to finish. Highly entertaining!",
        "Spectacular performances and breathtaking cinematography.",
        "An emotional rollercoaster that left me in tears of joy.",
        "Perfect blend of action, drama, and humor. Loved it!",
        "Innovative storytelling with remarkable character development.",
        "Visually stunning with an incredibly moving soundtrack.",
        "This movie is a work of art. Absolutely magnificent!",
        "Brilliant writing and exceptional directing. Five stars!",
        "Heart-warming story with outstanding performances.",
        "Engaging plot with unexpected twists. Thoroughly enjoyed!",
        "A cinematic gem that deserves all the praise it gets.",
        "Wonderful film with great message and superb acting.",
        "Entertaining throughout with fantastic special effects.",
        "Remarkable storytelling that keeps you hooked till the end.",
        "Beautiful film with incredible depth and emotion.",
        "Outstanding movie that combines great story with amazing visuals.",
        "Perfectly executed with brilliant performances by all actors.",
        "This film is pure magic! Absolutely loved every second.",
        "Incredible journey with stunning visuals and great music.",
        "Masterfully crafted with attention to every detail.",
        "Exceptional movie that touches your heart and soul.",
        "Brilliant concept executed flawlessly. Highly recommend!",
        "Amazing film with perfect pacing and great character arcs.",
        "Spectacular movie that sets new standards in filmmaking.",
        "Gorgeous cinematography combined with powerful storytelling.",
        "This movie is absolutely perfect in every way possible!"
    ]
    
    negative_reviews = [
        "Terrible movie, waste of time and money.",
        "Boring and predictable. Not recommended.",
        "Very disappointing. Poor storyline.",
        "Not worth watching. Very poor quality.",
        "Awful movie. Complete disaster.",
        "Very bad acting and direction.",
        "Worst movie I've ever seen.",
        "Completely boring and uninteresting.",
        "Poorly written script and bad direction.",
        "Absolute waste of time. Terrible in every way.",
        "Completely uninspiring and poorly executed.",
        "Terrible acting and confusing plot. Avoid at all costs!",
        "Boring storyline with wooden performances throughout.",
        "Disappointing film that fails to deliver on its promises.",
        "Poor direction and terrible screenplay. Very frustrating!",
        "Awful movie with no redeeming qualities whatsoever.",
        "Completely pointless film with terrible pacing.",
        "Bad acting, worse direction, and absolutely no plot.",
        "This movie was painful to watch. Truly horrible!",
        "Terrible waste of talent and completely boring.",
        "Poorly made film with no coherent storyline.",
        "Awful dialogue and unconvincing performances.",
        "This movie is a complete mess from start to finish.",
        "Boring, predictable, and poorly acted throughout.",
        "Terrible film that insults the audience's intelligence.",
        "Completely disappointing with no entertainment value.",
        "Poor quality production with amateur-level acting.",
        "This movie fails on every possible level.",
        "Absolutely terrible with no redeeming features.",
        "Boring plot and terrible character development.",
        "Poorly written with unconvincing performances.",
        "This film is a complete waste of time and money.",
        "Terrible movie that makes no sense at all.",
        "Boring and completely predictable storyline.",
        "Poor acting and even worse direction.",
        "This movie is painfully bad and boring.",
        "Terrible film with no entertainment value.",
        "Completely disappointing and poorly made.",
        "Awful movie that fails to engage the audience.",
        "This is possibly the worst movie ever made!"
    ]
    
    # Create balanced dataset
    all_reviews = positive_reviews + negative_reviews
    sentiments = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    return pd.DataFrame({'review': all_reviews, 'sentiment': sentiments})

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def train_improved_model():
    print("🎬 Training Improved Sentiment Analysis Model...")
    
    # Create larger dataset
    df = create_larger_dataset()
    print(f"📊 Created dataset with {len(df)} reviews ({len(df[df['sentiment']==1])} positive, {len(df[df['sentiment']==0])} negative)")
    
    # Preprocess text
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    print(f"🔄 Training on {len(X_train)} reviews, testing on {len(X_test)} reviews")
    
    # Create improved pipeline with better parameters
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore rare words
            max_df=0.8  # Ignore very common words
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            C=1.0,
            max_iter=1000
        ))
    ])
    
    # Train model
    print("🤖 Training improved model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
    
    # Detailed evaluation
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Test with challenging examples
    print("\n🧪 Testing with challenging reviews:")
    test_reviews = [
        "This movie is absolutely fantastic and amazing!",
        "Terrible film, completely boring and awful.",
        "The movie was okay, nothing special but watchable.",
        "Incredible cinematography but weak storyline.",
        "Not the best movie but has some good moments."
    ]
    
    predictions = model.predict(test_reviews)
    probabilities = model.predict_proba(test_reviews)
    
    for review, pred, prob in zip(test_reviews, predictions, probabilities):
        sentiment = "😊 Positive" if pred == 1 else "😞 Negative"
        confidence = max(prob) * 100
        print(f"  '{review}'")
        print(f"    → {sentiment} (Confidence: {confidence:.1f}%)")
        print()
    
    # Save model
    print("💾 Saving improved model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sentiment_model.pkl')
    print("✅ Improved model saved as models/sentiment_model.pkl")
    
    return model

if __name__ == "__main__":
    model = train_improved_model()
    print("\n🎉 Improved Sentiment Analysis Model Ready!")
    print("🚀 This model should now have much higher confidence scores!")