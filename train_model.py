import pandas as pd
import numpy as np
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by removing punctuation, converting to lowercase,
    removing numbers, and filtering out stopwords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def load_and_prepare_data():
    """
    Load the SMS dataset and prepare it for training
    """
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
    
    print(f"Dataset loaded: {len(df)} messages")
    print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
    print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
    
    # Add message length feature
    df['length'] = df['message'].apply(len)
    
    # Preprocess messages
    print("Preprocessing messages...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    return df

def create_model():
    """
    Create and return the machine learning pipeline
    """
    # Create pipeline with CountVectorizer, TF-IDF, and Naive Bayes
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    return pipeline

def train_model(df):
    """
    Train the model and return performance metrics
    """
    print("Training model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    print("\nModel Performance:")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return model, X_test, y_test, y_pred

def save_model(model, filename='spam_classifier_model.pkl'):
    """
    Save the trained model to a pickle file
    """
    print(f"\nSaving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

def test_model(model):
    """
    Test the model with some sample messages
    """
    print("\nTesting model with sample messages...")
    
    test_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "Ok lar... Joking wif u oni...",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Hello, how are you doing today?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net"
    ]
    
    for i, message in enumerate(test_messages, 1):
        processed_message = preprocess_text(message)
        prediction = model.predict([processed_message])[0]
        probability = model.predict_proba([processed_message])[0]
        
        print(f"\nTest {i}:")
        print(f"Message: {message[:100]}...")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probability)*100:.2f}%")

def main():
    """
    Main function to run the complete training pipeline
    """
    print("SMS Spam Classifier - Model Training")
    print("=" * 50)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Train model
        model, X_test, y_test, y_pred = train_model(df)
        
        # Save model
        save_model(model)
        
        # Test model
        test_model(model)
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("Model saved as 'spam_classifier_model.pkl'")
        print("You can now run the Streamlit app with: streamlit run app.py")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 