import praw
from datetime import datetime, timedelta
from datetime import timezone
import json
import os

reddit = praw.Reddit(user_agent=True, client_id='UXgY_9CupxBnlXV95PDOwA',
                     client_secret='GBKRhUWY7HV423A2J-6nN4SJrnVS1g',
                     username='DisastrousFeature830', password='Lxy12345')


subreddits = ['mentalhealth', 'depression', 'Anxiety', 'SuicideWatch', 'offmychest']

for subreddit_name in subreddits:
    os.makedirs(f"{subreddit_name}_posts", exist_ok=True)

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)

    count = 0
    idx = 0
    start_date = datetime(2024, 5, 28, tzinfo=timezone.utc)
    end_date = datetime.utcnow().replace(tzinfo=timezone.utc)

    for post in subreddit.top(limit=1000, time_filter='all'):
        post_time = datetime.utcfromtimestamp(post.created_utc).replace(tzinfo=timezone.utc)
        if start_date <= post_time <= end_date:
            post_info = {
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext,
                "created_utc": post.created_utc,
                "score": post.score,
                "num_comments": post.num_comments,
                "url": post.url,
                "comments": []
            }

            post.comments.replace_more(limit=0)
            for i, comment in enumerate(post.comments.list()):
                post_info["comments"].append({
                    "id": comment.id,
                    "body": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score
                })

            with open(f"{subreddit}_posts/reddit_post_{post.id}.json", "w", encoding="utf-8") as f:
                json.dump(post_info, f, ensure_ascii=False, indent=2)