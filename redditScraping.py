
import praw
import pandas as pd

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