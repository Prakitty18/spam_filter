#!/usr/bin/env python3
"""
Test script for SMS Spam Classifier Streamlit App
This script tests the core functionality before deployment
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def test_data_loading():
    """Test if the dataset can be loaded correctly"""
    print("Testing data loading...")
    try:
        df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
        print(f"✅ Dataset loaded successfully: {len(df)} messages")
        print(f"   - Spam messages: {len(df[df['label'] == 'spam'])}")
        print(f"   - Ham messages: {len(df[df['label'] == 'ham'])}")
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_model_training():
    """Test if the model can be trained"""
    print("\nTesting model training...")
    try:
        # Load data
        df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
        
        # Simple preprocessing
        df['processed_message'] = df['message'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        # Create and train model
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=1000)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        
        # Train on small subset for testing
        sample_size = min(1000, len(df))
        X_sample = df['processed_message'].head(sample_size)
        y_sample = df['label'].head(sample_size)
        
        pipeline.fit(X_sample, y_sample)
        
        # Test prediction
        test_message = "Free entry in 2 a wkly comp to win FA Cup final tkts"
        prediction = pipeline.predict([test_message])[0]
        probability = pipeline.predict_proba([test_message])[0]
        
        print(f"✅ Model trained successfully")
        print(f"   - Test prediction: {prediction}")
        print(f"   - Confidence: {max(probability)*100:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'nltk', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies available")
        return True

def test_file_structure():
    """Test if all required files are present"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'train_model.py',
        'requirements.txt',
        'README.md',
        'SMSSpamCollection'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All required files present")
        return True

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    print("\nTesting Streamlit app...")
    try:
        # Test if app.py can be imported
        import app
        print("✅ Streamlit app can be imported")
        return True
    except Exception as e:
        print(f"❌ Error importing Streamlit app: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 SMS Spam Classifier - Pre-deployment Tests")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Data Loading", test_data_loading),
        ("Model Training", test_model_training),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your app is ready for deployment.")
        print("\nNext steps:")
        print("1. Run: python train_model.py")
        print("2. Test locally: streamlit run app.py")
        print("3. Deploy to GitHub and Streamlit Cloud")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues before deployment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 