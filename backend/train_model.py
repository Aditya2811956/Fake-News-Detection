import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os

def preprocess_text(text):
    """Clean and preprocess the input text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset():
    """Load dataset from CSV files"""
    
    # Check if Kaggle dataset files exist
    if os.path.exists('Fake.csv') and os.path.exists('True.csv'):
        print("Loading Kaggle dataset...")
        
        # Load fake and real news
        fake_df = pd.read_csv('Fake.csv')
        real_df = pd.read_csv('True.csv')
        
        # Add labels
        fake_df['label'] = 0  # 0 = fake
        real_df['label'] = 1  # 1 = real
        
        # Combine title and text for better prediction
        fake_df['text'] = fake_df['title'] + ' ' + fake_df['text']
        real_df['text'] = real_df['title'] + ' ' + real_df['text']
        
        # Keep only text and label columns
        fake_df = fake_df[['text', 'label']]
        real_df = real_df[['text', 'label']]
        
        # Combine both datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Use subset for faster training
        df = df.head(5000)
        
        print(f"✓ Loaded {len(df)} articles from Kaggle dataset")
        return df
        
    else:
        print("⚠ Kaggle dataset not found in backend folder!")
        print("Please download Fake.csv and True.csv from Kaggle and place them in the backend folder")
        print("\nUsing sample dataset for now...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset for training (fallback option)"""
    # Sample fake news
    fake_news = [
        "Shocking discovery: Scientists prove earth is flat",
        "Breaking: Aliens landed in New York City yesterday",
        "Miracle cure found: This one weird trick cures all diseases",
        "Government hiding truth about mind control chips in vaccines",
        "Celebrity secretly controls world economy from underground base",
        "Unbelievable: Water found to be deadly poison by new study",
        "Exclusive: Time travel proven possible by unknown scientist",
        "Breaking news: Moon landing was completely fake and staged",
        "Shocking revelation: Birds are actually government drones",
        "Scientists baffled: Man survives without eating for 10 years",
    ]
    
    # Sample real news
    real_news = [
        "Stock market shows moderate gains in morning trading session",
        "Local school district announces new curriculum for next year",
        "Weather forecast predicts rain for the upcoming weekend",
        "City council approves budget for infrastructure improvements",
        "Research team publishes findings on climate change effects",
        "Technology company announces quarterly earnings report today",
        "Sports team wins championship after close final game",
        "University researchers develop new renewable energy method",
        "Government announces policy changes for healthcare system",
        "International summit addresses global economic challenges",
    ]
    
    # Create DataFrame
    data = pd.DataFrame({
        'text': fake_news + real_news,
        'label': [0] * len(fake_news) + [1] * len(real_news)  # 0 = fake, 1 = real
    })
    
    print(f"Created sample dataset with {len(data)} articles")
    return data

def train_model():
    """Train the fake news detection model"""
    print("=" * 50)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 50)
    
    # Load dataset
    df = load_dataset()
    
    # Preprocess text
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print("Training model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Save model and vectorizer
    print("\nSaving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model saved successfully!")
    print("\nYou can now run the Flask API using: python app.py")

if __name__ == '__main__':
    train_model()
