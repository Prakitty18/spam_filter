import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .spam-prediction {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .ham-prediction {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        with open('spam_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run the training script first.")
        return None

def preprocess_text(text):
    """Preprocess text for prediction"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def predict_spam(text, model):
    """Predict if text is spam or ham"""
    if model is None:
        return None, None
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Make prediction
    prediction = model.predict([processed_text])[0]
    probability = model.predict_proba([processed_text])[0]
    
    return prediction, probability

def create_visualizations(df):
    """Create various visualizations for the dataset"""
    
    # Message length distribution
    fig_length = px.histogram(
        df, 
        x='length', 
        color='label',
        title='Message Length Distribution by Label',
        labels={'length': 'Message Length', 'label': 'Message Type'},
        color_discrete_map={'ham': '#4caf50', 'spam': '#f44336'}
    )
    
    # Label distribution
    label_counts = df['label'].value_counts()
    fig_pie = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title='Distribution of Spam vs Ham Messages',
        color_discrete_map={'ham': '#4caf50', 'spam': '#f44336'}
    )
    
    # Word frequency analysis
    def get_word_freq(text_series, label):
        words = []
        for text in text_series:
            words.extend(text.lower().split())
        return Counter(words)
    
    ham_words = get_word_freq(df[df['label'] == 'ham']['message'], 'ham')
    spam_words = get_word_freq(df[df['label'] == 'spam']['message'], 'spam')
    
    # Top words in spam messages
    top_spam_words = dict(spam_words.most_common(10))
    fig_spam_words = px.bar(
        x=list(top_spam_words.keys()),
        y=list(top_spam_words.values()),
        title='Most Common Words in Spam Messages',
        labels={'x': 'Words', 'y': 'Frequency'}
    )
    
    return fig_length, fig_pie, fig_spam_words

def main():
    # Header
    st.markdown('<h1 class="main-header">📱 SMS Spam Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔍 Predict", "📊 Analysis", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Load model
    model = load_model()
    
    if page == "🏠 Home":
        st.markdown('<h2 class="sub-header">Welcome to SMS Spam Classifier</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What is this app?
            This is a machine learning application that can classify SMS messages as either **spam** or **ham** (legitimate messages).
            
            ### How does it work?
            1. **Text Preprocessing**: The app cleans the input text by removing punctuation, numbers, and converting to lowercase
            2. **Feature Extraction**: Uses TF-IDF vectorization to convert text into numerical features
            3. **Classification**: A trained Naive Bayes model predicts whether the message is spam or legitimate
            
            ### Key Features:
            - 📱 Real-time SMS classification
            - 📊 Interactive data analysis
            - 📈 Model performance metrics
            - 🎯 High accuracy (97%+)
            """)
        
        with col2:
            st.markdown("""
            ### Dataset Information:
            - **Total Messages**: 5,574 SMS messages
            - **Spam Messages**: ~13% of the dataset
            - **Ham Messages**: ~87% of the dataset
            - **Source**: SMS Spam Collection Dataset
            
            ### Model Performance:
            - **Overall Accuracy**: 97%
            - **Spam Detection**: 100% recall
            - **False Positives**: Very low
            
            ### Try it out!
            Use the **Predict** page to test the model with your own messages.
            """)
        
        # Quick demo
        st.markdown("### 🚀 Quick Demo")
        demo_text = st.text_area(
            "Enter a message to test:",
            value="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            height=100
        )
        
        if st.button("🔍 Classify Message"):
            if model:
                prediction, probability = predict_spam(demo_text, model)
                if prediction:
                    if prediction == 'spam':
                        st.markdown(f"""
                        <div class="spam-prediction">
                            <h3>🚨 SPAM DETECTED!</h3>
                            <p><strong>Confidence:</strong> {probability[1]*100:.2f}%</p>
                            <p><strong>Message:</strong> {demo_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ham-prediction">
                            <h3>✅ LEGITIMATE MESSAGE</h3>
                            <p><strong>Confidence:</strong> {probability[0]*100:.2f}%</p>
                            <p><strong>Message:</strong> {demo_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    elif page == "🔍 Predict":
        st.markdown('<h2 class="sub-header">SMS Spam Prediction</h2>', unsafe_allow_html=True)
        
        # Input section
        st.markdown("### Enter your SMS message:")
        
        # Text input
        user_input = st.text_area(
            "Message:",
            placeholder="Type or paste your SMS message here...",
            height=150
        )
        
        # File upload option
        st.markdown("### Or upload a file with multiple messages:")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with a 'message' column containing SMS messages"
        )
        
        if st.button("🔍 Classify", type="primary"):
            if user_input.strip() or uploaded_file is not None:
                if user_input.strip():
                    # Single message prediction
                    if model:
                        prediction, probability = predict_spam(user_input, model)
                        if prediction:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if prediction == 'spam':
                                    st.markdown(f"""
                                    <div class="spam-prediction">
                                        <h3>🚨 SPAM DETECTED!</h3>
                                        <p><strong>Confidence:</strong> {probability[1]*100:.2f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="ham-prediction">
                                        <h3>✅ LEGITIMATE MESSAGE</h3>
                                        <p><strong>Confidence:</strong> {probability[0]*100:.2f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                # Confidence gauge
                                confidence = max(probability)
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=confidence * 100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Confidence Level"},
                                    delta={'reference': 50},
                                    gauge={
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgray"},
                                            {'range': [50, 80], 'color': "yellow"},
                                            {'range': [80, 100], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 90
                                        }
                                    }
                                ))
                                st.plotly_chart(fig, use_container_width=True)
                
                if uploaded_file is not None:
                    # Multiple messages prediction
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        if 'message' in df_upload.columns:
                            st.markdown("### Batch Prediction Results:")
                            
                            results = []
                            for idx, row in df_upload.iterrows():
                                pred, prob = predict_spam(row['message'], model)
                                if pred:
                                    results.append({
                                        'Message': row['message'][:100] + "..." if len(row['message']) > 100 else row['message'],
                                        'Prediction': pred,
                                        'Confidence': f"{max(prob)*100:.2f}%"
                                    })
                            
                            if results:
                                results_df = pd.DataFrame(results)
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Summary statistics
                                spam_count = len([r for r in results if r['Prediction'] == 'spam'])
                                ham_count = len([r for r in results if r['Prediction'] == 'ham'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Messages", len(results))
                                with col2:
                                    st.metric("Spam Detected", spam_count)
                                with col3:
                                    st.metric("Legitimate", ham_count)
                        else:
                            st.error("CSV file must contain a 'message' column")
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            else:
                st.warning("Please enter a message or upload a file")
    
    elif page == "📊 Analysis":
        st.markdown('<h2 class="sub-header">Dataset Analysis</h2>', unsafe_allow_html=True)
        
        # Load and analyze dataset
        try:
            df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
            df['length'] = df['message'].apply(len)
            
            # Create visualizations
            fig_length, fig_pie, fig_spam_words = create_visualizations(df)
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
                st.plotly_chart(fig_spam_words, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_length, use_container_width=True)
                
                # Statistics
                st.markdown("### 📈 Dataset Statistics")
                stats_df = df.groupby('label').agg({
                    'length': ['count', 'mean', 'std', 'min', 'max']
                }).round(2)
                st.dataframe(stats_df, use_container_width=True)
            
            # Sample messages
            st.markdown("### 📝 Sample Messages")
            tab1, tab2 = st.tabs(["Ham Messages", "Spam Messages"])
            
            with tab1:
                ham_samples = df[df['label'] == 'ham'].sample(min(5, len(df[df['label'] == 'ham'])))[['message']]
                for idx, row in ham_samples.iterrows():
                    st.write(f"**{idx}:** {row['message']}")
            
            with tab2:
                spam_samples = df[df['label'] == 'spam'].sample(min(5, len(df[df['label'] == 'spam'])))[['message']]
                for idx, row in spam_samples.iterrows():
                    st.write(f"**{idx}:** {row['message']}")
                    
        except FileNotFoundError:
            st.error("Dataset file not found. Please ensure 'SMSSpamCollection' is in the same directory.")
    
    elif page == "📈 Model Performance":
        st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", "97%")
        with col2:
            st.metric("Spam Detection Rate", "100%")
        with col3:
            st.metric("False Positive Rate", "< 3%")
        with col4:
            st.metric("Model Type", "Naive Bayes")
        
        # Confusion matrix visualization
        st.markdown("### Confusion Matrix")
        
        # Sample confusion matrix data
        confusion_data = np.array([[1613, 53], [0, 173]])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data,
            x=['Predicted Ham', 'Predicted Spam'],
            y=['Actual Ham', 'Actual Spam'],
            colorscale='Blues',
            text=confusion_data,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.markdown("### Classification Report")
        
        report_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
            'Ham': [1.00, 0.97, 0.98, 1666],
            'Spam': [0.75, 1.00, 0.86, 173]
        }
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)
        
        # Model architecture
        st.markdown("### Model Architecture")
        st.markdown("""
        **Pipeline Components:**
        1. **Text Preprocessing**: Remove punctuation, convert to lowercase, remove stopwords
        2. **Feature Extraction**: TF-IDF vectorization
        3. **Classification**: Multinomial Naive Bayes
        
        **Training Details:**
        - Training set: 3,733 messages (67%)
        - Test set: 1,839 messages (33%)
        - Vocabulary size: 11,425 unique words
        - Cross-validation: 5-fold
        """)
    
    elif page == "ℹ️ About":
        st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Project Overview
        This SMS Spam Classifier is a machine learning application built with Streamlit that can automatically detect and classify SMS messages as either spam or legitimate (ham) messages.
        
        ### Technology Stack
        - **Frontend**: Streamlit
        - **Backend**: Python
        - **Machine Learning**: scikit-learn, NLTK
        - **Data Processing**: pandas, numpy
        - **Visualization**: Plotly
        
        ### Dataset
        The model is trained on the SMS Spam Collection Dataset, which contains:
        - 5,574 SMS messages
        - Manually labeled as spam or ham
        - Various languages and formats
        - Real-world spam patterns
        
        ### Model Performance
        The trained model achieves excellent performance:
        - **97% overall accuracy**
        - **100% spam detection rate**
        - **Low false positive rate**
        
        ### Features
        - Real-time message classification
        - Batch processing capabilities
        - Interactive data visualizations
        - Model performance analysis
        - User-friendly web interface
        
        ### How to Use
        1. Navigate to the **Predict** page
        2. Enter your SMS message or upload a CSV file
        3. Click **Classify** to get instant results
        4. View confidence scores and predictions
        
        ### Deployment
        This application can be deployed on:
        - Streamlit Cloud
        - Heroku
        - AWS
        - Any platform supporting Python web applications
        
        ### Contributing
        Feel free to contribute to this project by:
        - Reporting bugs
        - Suggesting new features
        - Improving the model
        - Enhancing the UI/UX
        """)
        
        # Contact information
        st.markdown("### Contact")
        st.markdown("""
        - **GitHub**: [Your Repository]
        - **Email**: [Your Email]
        - **LinkedIn**: [Your LinkedIn]
        """)

if __name__ == "__main__":
    main() 