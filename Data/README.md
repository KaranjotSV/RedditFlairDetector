# Data

The features retrieved for each post are **Post ID**, **Title**, **URL**, **Body**, **Score**, **Comments**(Top 10), **Comments Count**, **Time Stamp**, **Flair** 

- [PRAW-Data.csv](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/Data/PRAW-Data.csv) contains 2656 examples of posts retrieved using Python Reddit API Wrapper — PRAW, 
the data contains 235 - 242 examples of each Flair. The method used to retrieve is **subreddit.search(f"flair:{flair}", limit = 300)**,
the limit for each search is set to 300. An alternative method is **subreddit.search("flair", limit = 300)**, but it searches the "flair"
in **post.title** instead of **post.link_flair_text**, this leads to retrieval of mislabelled data.

- [PRAW-Preprocessed.csv](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/Data/PRAW-Preprocessed.csv) is the preprocessed 
version of [PRAW-Data.csv](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/Data/PRAW-Data.csv), obtained after 
Exploratory Data Analysis.

- [PRAW-CombinedFeature.csv](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/Data/PRAW-CombinedFeature.csv) contains 
5312 examples, of which the first 2656 examples are formed by concatenating **Title**, **Body**, **Comments** of each example 
in [PRAW-Preprocessed.csv](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/Data/PRAW-Preprocessed.csv), the other 
2656 examples are formed by augmenting each example of the first 2656 examples, this was done to increase the total number of 
examples, in order to reduce over-fitting.
