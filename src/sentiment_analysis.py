import json
import pandas as pd
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import pipeline
import os

model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
distil_bert_model = pipeline(task="sentiment-analysis", model=model_checkpoint)


def get_sentiment(input_folder, input_file, distil_bert_model=distil_bert_model):
    input_path = f"{input_folder}/{input_file}"
    with open(input_path, 'r', encoding='utf-8') as jsonf:
        data = json.load(jsonf)
        
        df = {}
        df['id'] = data['id']
        df['subreddit'] = input_folder.split('_')[0]
        df['title'] = data['title']
        df['time'] = data['created_utc']
        # Get the sentiment for the 'selftext' field
        try:
            df['sentiment_selftext'] = distil_bert_model(data['selftext'])
            score = df['sentiment_selftext'][0].get('score', None)
            label = df['sentiment_selftext'][0].get('label', None)
            df['sentiment_selftext_label'] = label
            df['sentiment_selftext_score'] = -score if label == 'NEGATIVE' else score
        except Exception as e:
            print(f"Error processing selftext for {input_file} in {input_folder}: {e}")

        # Get the sentiment for the 'body' of each comment if they exist
        if 'comments' in data:
            positive_comment_scores = []
            negative_comment_scores = []
            for comment in data['comments']:
                # print(comment['id'])
                try:
                    comment_sentiment = distil_bert_model(comment['body'])
                except Exception as e:
                    print(f"Error processing comment {comment['id']} from {input_path}: {e}")
                    continue
                if comment_sentiment[0].get('label', None) == 'POSITIVE':
                    positive_comment_scores.append(comment_sentiment[0].get('score', None))
                else:
                    negative_comment_scores.append(comment_sentiment[0].get('score', None))
            df['comment_positive_avg_score'] = sum(positive_comment_scores) / len(positive_comment_scores) if positive_comment_scores else None
            df['comment_negative_avg_score'] = sum(negative_comment_scores) / len(negative_comment_scores) if negative_comment_scores else None
            df['comment_label'] = 'POSITIVE' if df['comment_positive_avg_score'] > df['comment_negative_avg_score'] else 'NEGATIVE' if df['comment_positive_avg_score'] < df['comment_negative_avg_score'] else 'NEUTRAL'
            df['comment_score'] = df['comment_positive_avg_score'] if df['comment_label'] == 'POSITIVE' else -sum(negative_comment_scores) / len(negative_comment_scores) if df['comment_label'] == 'NEGATIVE' else df['comment_positive_avg_score']
        else:
            df['comment_negative_avg_score'] = None
            df['comment_positive_avg_score'] = None
            df['comment_label'] = None
            df['comment_score'] = None
        
        df = pd.DataFrame([df])
        return df
    
def convert_to_df():
    input_folders = ['mentalhealth_posts_cleaned', 'depression_posts_cleaned', 'Anxiety_posts_cleaned', 'SuicideWatch_posts_cleaned', 'offmychest_posts_cleaned']
    # input_folders = ['offmychest_posts_cleaned']
    for input_dir in input_folders:
        print(f"Input folder: {input_dir}")
        
        df = pd.DataFrame()
        for filename in os.listdir(input_dir):
            df = pd.concat([df, get_sentiment(input_dir, filename)], ignore_index=True)
        df.to_csv(f'{input_dir}_sentiment_analysis.csv', index=False)

if __name__ == "__main__":
    convert_to_df()

# print(get_sentiment('depression_posts_cleaned', 'cleaned_reddit_post_1g9mnuy.json')[['sentiment_selftext_label', 'sentiment_selftext_score', 'comment_label', 'comment_score']].head())