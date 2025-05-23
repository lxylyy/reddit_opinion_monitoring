# Reddit Mental Health Analyzer

A Streamlit web app to fetch posts from mental health-related subreddits, perform sentiment analysis (VADER, BERT), topic modeling (LDA, BERT), and visualize the results.

## Features
- Fetch posts from r/mentalhealth, r/depression, r/Anxiety
- Sentiment analysis using VADER and BERT
- Topic modeling using LDA and BERT embeddings
- Interactive visualizations

## Setup Instructions

### 1. Clone the Repository
```bash
# If you haven't already
cd /path/to/your/project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Reddit API Credentials
1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Set:
   - **name**: Any name (e.g., MentalHealthAnalyzer)
   - **type**: script
   - **redirect uri**: http://localhost:8080
4. After creation, copy your **client_id** and **client_secret**
5. Set a **user_agent** (e.g., `MentalHealthAnalyzer/0.1 by YOUR_REDDIT_USERNAME`)
6. Open `app.py` and fill in:
   ```python
   REDDIT_CLIENT_ID = 'YOUR_CLIENT_ID'
   REDDIT_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
   REDDIT_USER_AGENT = 'MentalHealthAnalyzer/0.1 by YOUR_REDDIT_USERNAME'
   ```

### 4. Run the App
```bash
streamlit run app.py
```

## Usage
- Select subreddits, number of posts, and analysis types in the sidebar
- Click "Fetch and Analyze"
- View sentiment and topic modeling results with interactive plots

## Troubleshooting
- **ModuleNotFoundError**: Make sure you installed requirements in the correct Python environment
- **Reddit API errors**: Double-check your credentials in `app.py`
- **No posts found**: Try increasing the number of posts or check your internet connection
- **Long processing time**: BERT-based analysis and topic modeling can be slow for large datasets

## License
MIT 