import streamlit as st
import praw
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import plotly.express as px
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# --- INSTRUCTIONS FOR REDDIT API CREDENTIALS ---
# 1. Go to https://www.reddit.com/prefs/apps
# 2. Create a new 'script' app and get your client_id, client_secret, and set a user_agent (any string)
# 3. Fill them below:
REDDIT_CLIENT_ID = 'MYBnuzkyiy4hmdtDyk5m4w'
REDDIT_CLIENT_SECRET = 'Iptii4SvNTaUNbM3TrWCIkTqC5G_2Q'
REDDIT_USER_AGENT = 'MentalHealthAnalyzer/0.1'

# --- NLTK Downloads ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- SIDEBAR ---
st.sidebar.title('Reddit Mental Health Analyzer')
subreddits = st.sidebar.multiselect('Choose subreddits:', ['r/mentalhealth', 'r/depression', 'r/Anxiety'], default=['r/mentalhealth'])
num_posts = st.sidebar.slider('Number of posts per subreddit', 10, 100, 20)
analysis_type = st.sidebar.multiselect('Analysis Type', ['Sentiment Analysis (VADER)', 'Sentiment Analysis (BERT)', 'Topic Modeling (LDA)', 'Topic Modeling (BERT)'], default=['Sentiment Analysis (VADER)', 'Topic Modeling (LDA)'])

st.title('Reddit Mental Health Posts Analyzer')

# --- REDDIT FETCHING ---
def fetch_reddit_posts(subreddits, num_posts):
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)
    posts = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub.replace('r/', ''))
        for post in subreddit.hot(limit=num_posts):
            if not post.stickied:
                posts.append({
                    'subreddit': sub,
                    'title': post.title,
                    'text': post.selftext,
                    'full_text': post.title + ' ' + post.selftext
                })
    return pd.DataFrame(posts)

# --- TEXT PREPROCESSING ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def preprocess_df(df):
    df['clean_text'] = df['full_text'].apply(preprocess)
    return df

# --- SENTIMENT ANALYSIS ---
def vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['vader_compound'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['vader_sentiment'] = df['vader_compound'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    return df

def bert_sentiment(df):
    bert_pipe = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    df['bert_sentiment'] = df['clean_text'].apply(lambda x: bert_pipe(x[:512])[0]['label'])
    return df

# --- TOPIC MODELING (LDA) ---
def lda_topic_modeling(df, num_topics=5):
    texts = [t.split() for t in df['clean_text']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    topics = lda_model.print_topics(num_words=5)
    # Assign dominant topic to each doc
    df['lda_topic'] = [max(lda_model[doc], key=lambda x: x[1])[0] if lda_model[doc] else -1 for doc in corpus]
    return df, lda_model, corpus, dictionary, topics

# --- TOPIC MODELING (BERT + KMeans) ---
def bert_topic_modeling(df, num_topics=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['clean_text'].tolist(), show_progress_bar=True)
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    df['bert_topic'] = clusters
    return df, kmeans, embeddings

# --- VISUALIZATION ---
def plot_sentiment(df, method):
    if method == 'VADER':
        fig = px.histogram(df, x='vader_sentiment', color='vader_sentiment', title='VADER Sentiment Distribution')
        st.plotly_chart(fig)
    elif method == 'BERT':
        fig = px.histogram(df, x='bert_sentiment', color='bert_sentiment', title='BERT Sentiment Distribution')
        st.plotly_chart(fig)

def plot_lda_topics(df, topics):
    fig = px.histogram(df, x='lda_topic', color='lda_topic', title='LDA Topic Distribution', nbins=len(topics))
    st.plotly_chart(fig)
    st.write('LDA Topics:')
    for tid, topic in topics:
        st.write(f"Topic {tid}: {topic}")

def plot_bert_topics(df, num_topics):
    fig = px.histogram(df, x='bert_topic', color='bert_topic', title='BERT Topic Distribution', nbins=num_topics)
    st.plotly_chart(fig)

# --- MAIN APP LOGIC ---
if st.button('Fetch and Analyze'):
    if 'YOUR_CLIENT_ID' in REDDIT_CLIENT_ID or 'YOUR_CLIENT_SECRET' in REDDIT_CLIENT_SECRET:
        st.error('Please set your Reddit API credentials at the top of this file.')
        st.stop()
    st.write('Fetching posts from:', subreddits)
    df = fetch_reddit_posts(subreddits, num_posts)
    if df.empty:
        st.warning('No posts found. Try increasing the number of posts or check your credentials.')
        st.stop()
    st.success(f"Fetched {len(df)} posts.")
    st.write(df[['subreddit', 'title', 'text']])

    st.write('Preprocessing text...')
    df = preprocess_df(df)
    st.success('Text preprocessed.')

    # Sentiment Analysis
    if 'Sentiment Analysis (VADER)' in analysis_type:
        st.write('Running VADER sentiment analysis...')
        df = vader_sentiment(df)
        plot_sentiment(df, 'VADER')
    if 'Sentiment Analysis (BERT)' in analysis_type:
        st.write('Running BERT sentiment analysis...')
        df = bert_sentiment(df)
        plot_sentiment(df, 'BERT')

    # Topic Modeling
    if 'Topic Modeling (LDA)' in analysis_type:
        st.write('Running LDA topic modeling...')
        df, lda_model, corpus, dictionary, topics = lda_topic_modeling(df)
        plot_lda_topics(df, topics)
        # Interactive LDA visualization
        with st.expander('Show interactive LDA visualization'):
            with tempfile.TemporaryDirectory() as tmpdirname:
                vis = gensimvis.prepare(lda_model, corpus, dictionary)
                html_path = os.path.join(tmpdirname, 'lda.html')
                pyLDAvis.save_html(vis, html_path)
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_string = f.read()
                st.components.v1.html(html_string, height=800, scrolling=True)
    if 'Topic Modeling (BERT)' in analysis_type:
        st.write('Running BERT-based topic modeling...')
        num_topics = 5
        df, kmeans, embeddings = bert_topic_modeling(df, num_topics=num_topics)
        plot_bert_topics(df, num_topics)

    st.write('Analysis complete!')
else:
    st.info('Select options and click "Fetch and Analyze" to begin.') 