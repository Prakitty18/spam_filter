#!/usr/bin/env python3
"""
Setup script for SMS Spam Classifier Streamlit App
This script automates the initial setup process
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("This app requires Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    nltk_script = """
import nltk
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK stopwords already available")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("NLTK stopwords downloaded successfully")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", nltk_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading NLTK data: {e.stderr}")
        return False

def train_model():
    """Train the spam classification model"""
    print("\n🤖 Training the model...")
    
    if not run_command("python train_model.py", "Training model"):
        return False
    
    # Check if model file was created
    if os.path.exists('spam_classifier_model.pkl'):
        print("✅ Model file created successfully")
        return True
    else:
        print("❌ Model file not found after training")
        return False

def test_app():
    """Test if the app can be started"""
    print("\n🧪 Testing the app...")
    
    # Try to import the app
    try:
        import app
        print("✅ App can be imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing app: {e}")
        return False

def create_sample_csv():
    """Create a sample CSV file for testing batch processing"""
    print("\n📄 Creating sample CSV file...")
    
    sample_data = {
        'message': [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            "Ok lar... Joking wif u oni...",
            "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "Hello, how are you doing today?",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net"
        ]
    }
    
    try:
        import pandas as pd
        df = pd.DataFrame(sample_data)
        df.to_csv('sample_messages.csv', index=False)
        print("✅ Sample CSV file created: sample_messages.csv")
        return True
    except Exception as e:
        print(f"❌ Error creating sample CSV: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 SMS Spam Classifier - Setup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found in current directory")
        print("Please run this script from the project root directory")
        return 1
    
    # Run setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Download NLTK Data", download_nltk_data),
        ("Train Model", train_model),
        ("Test App", test_app),
        ("Create Sample CSV", create_sample_csv)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        result = step_func()
        results.append((step_name, result))
        
        if not result:
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please fix the issue and run the setup again")
            return 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{step_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} steps completed successfully")
    
    if passed == total:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Test the app locally:")
        print("   streamlit run app.py")
        print("\n2. Deploy to GitHub:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit'")
        print("   git remote add origin https://github.com/YOUR_USERNAME/sms-spam-classifier.git")
        print("   git push -u origin main")
        print("\n3. Deploy to Streamlit Cloud:")
        print("   - Go to https://share.streamlit.io")
        print("   - Connect your GitHub repository")
        print("   - Deploy the app")
        print("\n📚 For detailed instructions, see: deploy_instructions.md")
    else:
        print(f"\n⚠️  {total - passed} step(s) failed. Please fix the issues and run setup again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 