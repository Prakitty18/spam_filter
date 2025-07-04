# 🚀 Deployment Guide - SMS Spam Classifier

This guide will walk you through deploying your SMS Spam Classifier Streamlit app using GitHub and Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: You need a GitHub account
2. **Python Environment**: Local Python installation for testing
3. **Git**: Git installed on your local machine

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Local Environment

1. **Install dependencies locally first**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**:
   ```bash
   python train_model.py
   ```

3. **Test the app locally**:
   ```bash
   streamlit run app.py
   ```

4. **Verify everything works** at `http://localhost:8501`

### Step 2: Push to GitHub

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: SMS Spam Classifier Streamlit App"
   ```

2. **Create a new repository on GitHub**:
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name it `sms-spam-classifier` (or your preferred name)
   - Make it public
   - Don't initialize with README (we already have one)

3. **Connect and push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/sms-spam-classifier.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy your app**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/sms-spam-classifier`
   - Set the main file path: `app.py`
   - Click "Deploy"

3. **Wait for deployment**:
   - Streamlit will automatically install dependencies
   - The app will be available at `https://your-app-name.streamlit.app`

### Step 4: Verify Deployment

1. **Check the app URL** provided by Streamlit Cloud
2. **Test all features**:
   - Single message classification
   - Batch file upload
   - Data analysis pages
   - Model performance metrics

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Model file not found
**Error**: `FileNotFoundError: spam_classifier_model.pkl`

**Solution**: 
- Make sure you ran `python train_model.py` locally
- Check that `spam_classifier_model.pkl` is in your repository
- If the file is large, consider using Git LFS or hosting it elsewhere

#### Issue 2: NLTK data not found
**Error**: `LookupError: Resource stopwords not found`

**Solution**:
- The app automatically downloads NLTK data
- If it fails, you can manually download it:
  ```python
  import nltk
  nltk.download('stopwords')
  ```

#### Issue 3: Dependencies not installed
**Error**: `ModuleNotFoundError`

**Solution**:
- Check that `requirements.txt` is in your repository
- Verify all dependencies are listed correctly
- Streamlit Cloud will automatically install them

#### Issue 4: App crashes on startup
**Error**: Various startup errors

**Solution**:
- Check the Streamlit Cloud logs
- Ensure all files are properly committed to GitHub
- Test locally first to catch issues

### Performance Optimization

1. **Model Size**: The pickle file should be under 100MB for Streamlit Cloud
2. **Memory Usage**: Monitor memory usage in the Streamlit Cloud dashboard
3. **Response Time**: Consider caching expensive operations

## 📊 Monitoring Your App

### Streamlit Cloud Dashboard

1. **Access your dashboard** at [share.streamlit.io](https://share.streamlit.io)
2. **Monitor metrics**:
   - App uptime
   - Memory usage
   - Response times
   - Error rates

### Logs and Debugging

1. **View logs** in the Streamlit Cloud dashboard
2. **Check for errors** in the logs section
3. **Monitor performance** metrics

## 🔄 Updating Your App

### Making Changes

1. **Edit your code locally**
2. **Test changes** with `streamlit run app.py`
3. **Commit and push** to GitHub:
   ```bash
   git add .
   git commit -m "Update: [describe your changes]"
   git push origin main
   ```
4. **Streamlit Cloud will automatically redeploy**

### Version Control Best Practices

1. **Use meaningful commit messages**
2. **Test locally before pushing**
3. **Keep your repository clean**
4. **Document major changes**

## 🌐 Custom Domain (Optional)

If you want to use a custom domain:

1. **Purchase a domain** (e.g., from Namecheap, GoDaddy)
2. **Configure DNS** to point to your Streamlit app
3. **Contact Streamlit support** for custom domain setup

## 📈 Scaling Considerations

### For High Traffic

1. **Consider alternative hosting**:
   - Heroku
   - AWS
   - Google Cloud Platform
   - DigitalOcean

2. **Optimize your app**:
   - Cache expensive operations
   - Use efficient data structures
   - Minimize model size

### Cost Considerations

- **Streamlit Cloud**: Free tier available
- **Other platforms**: May have costs for higher usage

## 🎯 Next Steps

After successful deployment:

1. **Share your app** with others
2. **Collect feedback** and improve
3. **Add new features**:
   - User authentication
   - API endpoints
   - More visualization options
   - Model retraining capabilities

4. **Monitor usage** and performance
5. **Keep dependencies updated**

## 📞 Support

If you encounter issues:

1. **Check Streamlit documentation**: [docs.streamlit.io](https://docs.streamlit.io)
2. **Visit Streamlit community**: [discuss.streamlit.io](https://discuss.streamlit.io)
3. **GitHub issues**: Create an issue in your repository

---

🎉 **Congratulations! Your SMS Spam Classifier is now live on the web!** 