import praw
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import plotly.express as px
import os
import zipfile
from urllib.request import Request, urlopen

nltk.download('stopwords')

reddit = praw.Reddit(
    client_id="rVEWGIb3yoZbaogQwD355g",
    client_secret="YJWC0Ygh7eJRhOv9ow6kQ5uAZMhnmw",
    user_agent="AI_Sentiment_Analysis:v1.0 (by u/Personal_Sorbet_5319)",
    username="Personal_Sorbet_5319",
    password="STmonkey12!"
)

reddit_links = [
    "https://www.reddit.com/r/Cornell/comments/10e0wcw/do_you_plan_to_use_an_ai_chatbot_to_help_write/",
    "https://www.reddit.com/r/Cornell/comments/10m4l47/has_chatgpt_been_mentioned_in_anyone_elses_classes/",
    "https://www.reddit.com/r/NoStupidQuestions/comments/1g7e5l6/will_ai_make_education_academics_careers/",
    "https://www.reddit.com/r/education/comments/1ig4qo4/using_ai_as_a_student_in_2025/",
    "https://www.reddit.com/r/OpenAI/comments/1gqau32/education_ai_reversal_what_if_education_told/",
    "https://www.reddit.com/r/AskComputerScience/comments/1h3ia75/is_pursuing_a_cs_degree_still_worth_it_given_the/",
    "https://www.reddit.com/r/learnprogramming/comments/14n4u4j/my_computer_science_teachers_told_me_that_they_no/",
    "https://www.reddit.com/r/careeradvice/comments/1iu1cjs/is_studying_cs_still_worth_it/",
    "https://www.reddit.com/r/singularity/comments/11sy81y/what_are_the_best_careers_to_pursue_in_the_next/"
]

data = []
for url in reddit_links:
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments:
        if comment.body not in ["[removed]", "[deleted]"]:
            data.append({
                "post_title": submission.title,
                "comment_text": comment.body,
                "url": url
            })

df = pd.DataFrame(data)
df.to_csv("reddit_ai_sentiment.csv", index=False)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    blob = TextBlob(text)
    vader_score = analyzer.polarity_scores(text)['compound']
    return pd.Series({
        "textblob_polarity": blob.sentiment.polarity,
        "vader_compound": vader_score
    })

df = df.join(df['comment_text'].apply(get_sentiment_scores))

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['vader_compound'].apply(classify_sentiment)

def extract_sentiment_words(text):
    blob = TextBlob(text)
    positives = []
    negatives = []
    neutrals = []

    for word_group, polarity, _ in blob.sentiment_assessments.assessments:
        if polarity > 0.1:
            positives.extend(word_group)
        elif polarity < -0.1:
            negatives.extend(word_group)
        else:
            neutrals.extend(word_group)

    return pd.Series({
        "positive_words": positives,
        "negative_words": negatives,
        "neutral_words": neutrals
    })

df = df.join(df['comment_text'].apply(extract_sentiment_words))

df.to_csv("reddit_ai_sentiment_full.csv", index=False)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='sentiment', hue='sentiment', data=df, palette='Set2',
                   order=['Positive', 'Neutral', 'Negative'], legend=False)

plt.title("Reddit Sentiment Toward AI in Education", fontsize=16)
plt.xlabel("Sentiment Category", fontsize=13)
plt.ylabel("Number of Comments", fontsize=13)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# nrc_url = "https://www.saifmohammad.com/Lexicons/NRC-Emotion-Lexicon.zip"
# nrc_zip_path = "NRC-Emotion-Lexicon.zip"
# nrc_txt_filename = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# # Step 1: Download with custom headers if file not already extracted
# if not os.path.exists(nrc_txt_filename):
#     print("Downloading NRC Emotion Lexicon...")
#     req = Request(nrc_url, headers={'User-Agent': 'Mozilla/5.0'})
#     with urlopen(req) as response, open(nrc_zip_path, 'wb') as out_file:
#         out_file.write(response.read())

#     print("Extracting NRC Emotion Lexicon...")
#     with zipfile.ZipFile(nrc_zip_path, 'r') as zip_ref:
#         zip_ref.extractall()

# # Step 2: Load the lexicon
# nrc = pd.read_csv(nrc_txt_filename, sep='\t', header=None, names=["word", "emotion", "association"])
# nrc = nrc[nrc['association'] == 1][['word', 'emotion']]

# # Step 3: Count emotions in comments
# stop_words = set(stopwords.words("english"))
# emotion_counts = defaultdict(int)

# for comment in df['comment_text']:
#     words = comment.lower().split()
#     for word in words:
#         clean_word = word.strip(".,!?()[]{}<>\"'`~;:-").lower()
#         if clean_word in stop_words or len(clean_word) < 2:
#             continue
#         emotions = nrc[nrc['word'] == clean_word]['emotion'].tolist()
#         for emotion in emotions:
#             emotion_counts[emotion] += 1

# emotion_df = pd.DataFrame.from_dict(emotion_counts, orient='index', columns=['count']).reset_index()
# emotion_df.columns = ['emotion', 'count']
# emotion_df = emotion_df.sort_values(by="count", ascending=False)

# # --- Treemap of Emotion Distribution ---
# fig = px.treemap(emotion_df, path=['emotion'], values='count',
#                  title="Emotion Distribution in Reddit Comments (NRC Lexicon)",
#                  color='count', color_continuous_scale='RdBu')

# fig.write_html("emotion_treemap.html")
# fig.show()
