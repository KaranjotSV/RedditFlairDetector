# Reddit Flair Detector

Reddit Flair Detector is a web application which detects flairs of posts of subreddit india using Machine Learning algorithms. The application can be found live at [redditflairdetector-ksv](https://redditflairdetector-ksv.herokuapp.com/).

### Directory Structure

The description of files and folders can be found below:
  1. [Data-collection](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/Data-Collection) - Folder containing jupyter notebooks of data collection from r/india, contains 2 notebooks, one for each API - PRAW and PushShift.
  2. [Data](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/Data) - Folder containing CSV instances of the collected data.
  4. [Models](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/Models) - Folder containing jupyter notebooks of model training, contains 2 notebooks, each for the models fitted on the data retrieved via APIs - PRAW and PushShift .
  5. [sample](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/sample) - Folder containing a sample txt file,
the txt file contains URLs of subreddit india posts, can be used for testing [/automated_testing](https://redditflairdetector-ksv.herokuapp.com/automated_testing) endpoint of the web application endpoint of the web application.
  6. [static](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/static) - Folder containing CSS files.
  7. [templates](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/templates) - Folder containing HTML files.
  8. [EDA.ipynb](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/EDA.ipynb) - Jupyter notebook for Exploratory Data Analysis.
  9. [Procfile](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/Procfile) - Needed for setting up Heroku.
  10. [app.py](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/app.py) - Flask application of the model.
  11. [nltk.txt](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/nltk.txt) - Containing all NLTK library needed dependencies.
  12. [requirements.txt](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/requirements.txt) - Containing all Python dependencies of the project.

### Project Execution

  1. Open the `Terminal`.
  2. Clone the repository by entering `git clone https://github.com/KaranjotSV/RedditFlairDetector.git`.
  3. Ensure that `Python3` and `pip` is installed on the system.
  4. Create a `virtualenv` by executing the following command: `virtualenv -p python3 env`.
  5. Activate the `env` virtual environment by executing the follwing command: `source env/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Enter `python` shell and `import nltk`. Execute `nltk.download(dependency)` for dependencies listed in nltk.txt and exit the shell.
  8. Now, execute the following command: `python3 app.py` and it will point to the `localhost` with the port.
  9. Hit the `IP Address` on a web browser and use the application.
  
### Dependencies

The following dependencies can be found in [requirements.txt](https://github.com/KaranjotSV/RedditFlairDetector/blob/master/requirements.txt):

  1. [praw](https://praw.readthedocs.io/en/latest/)
  2. [scikit-learn](https://scikit-learn.org/)
  3. [nltk](https://www.nltk.org/)
  4. [pandas](https://pandas.pydata.org/)
  5. [numpy](http://www.numpy.org/)
  6. [flask](https://flask.palletsprojects.com/en/1.1.x/)
  7. [gunicorn](https://gunicorn.org/)
  
### Approach

The approach for this task is more focussed upon Data, rather than the Machine learning algorithms, such an approach is followed because of the Machine learning models overfitting the data. Various techniques like Data Augmentation, Oversampling, Feature Extraction are applied in order to get better results.

The approach is explained in detail as follows:

  1. Collected maximum possible(234 - 243 for each flair) India subreddit data for each of the 11 flairs using `praw` module [[1]](http://www.storybench.org/how-to-scrape-reddit-with-python/). The method used for searching posts for each flair is `subreddit.search(f"flair:{flair}", limit = 300)`, another alternative is `subreddit.search(flair, limit = 300)`, but this leads to collection of mislabelled data, as it searched the flair in `post.title` rather than `post.link_flair_text`.
  2. The data included ***Post ID, Title, URL, Body, Score, Comments, Comments Count, Time Stamp*** and ***Flair***.
  3. For ***Comments***, only top level comments(top 10) were considered in dataset and no sub-comments were considered.
  4. The ***Title, Comments*** and ***Body*** were cleaned by removing non-english words, stopwords and bad symbols using `nltk`.
  5. Five types of features were considered for the the given task:
  
    a) Title
    b) Comments
    c) Combining Title, Comments as one feature
    d) Combining Title, Comments, Body as one feature.
    
  6. The dataset was augmented to increase the number of examples, in order to avoid overfitting. Augmentation was done by shuffling the sentences and changing the words in the sentences with their synonyms, using `nltk`
  7. The dataset was splitted into **80% train** and **20% test** data using `train-test-split` of `scikit-learn`.
  8. The dataset was then converted into a `Vector` and `TF-IDF` form.
  9. Then, the following ML algorithms (using `scikit-learn` libraries) were applied on the dataset:
    
    a) Naive-Bayes
    b) Logistic Regression
    c) Random Forest
    
 10. Training and Testing on the dataset showed that **Logistic Regression** showed the best testing accuracy of **63%** when trained on the combination of **Title + Comments + Body** feature.
 11. The models were overfitting, even after tuning the hyperparameters, this led to the idea of collecting more data.
 12. Collected 24,000 India subreddit post for 11 flairs, posted between 16th February, 2019 and 24th April, 2020 using `pushshift` module. The posts were considered for the dataset only if, their titles consisted only english words.
 13. Same data cleaning and preprocessing methodoligies were applied, other than data augmentation, on data retrieved using `pushshift`
 13. Same Five types of features were considered for the task as the ones which were considered from data retrieved using `praw` module.
 14. The dataset was splitted into **80% train** and **20% test** data using `train-test-split` of `scikit-learn`.
 15. The dataset was then converted into a `Vector` and `TF-IDF` form.
 16. Due to imbalanced dataset, oversampling techniques were applied, among which SMOTE(Synthetic Minority Oversampling Technique) worked the best.
 17. Training and Testing on the dataset again showed that **Logistic Regression** showed the best testing accuracy of **54%** when trained on the combination of **Title + Comments + Body** feature.
 18. The model was saved and is being used for prediction of the flair from the URL of the post.
 19. This model was less accurate on the test data that the one trained on dataset retrieved using `praw`, but it was much less overfitted, and showed better results when tested on URLs of posts from subreddit India.
    
### Results

#### Title as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | **0.5360134003**  |
| Logistic Regression        | 0.5351758793      |
| Random Forest              | 0.5339195979      |

#### Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.2830820770      |
| Logistic Regression        | **0.4020100502**  |
| Random Forest              | 0.3890284757      |

#### Title + Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.4941373534      |
| Logistic Regression        | **0.5241678978**  |
| Random Forest              | 0.5197403685      |

#### Title + Comments + Body as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.4924623115      |
| Logistic Regression        | **0.5413567839**  |
| Random Forest              | 0.515912897823    |

### Comment on using PushShift module for data collection and accuracy of models

The approach for this task was more focussed upon data because of the over-fitting of machine learning models. The data collection task was shifted to `pushshift` module because of the limit set by `praw` module on number of posts that could be retrieved. Moreover, if the data collected using `praw` module by `subreddit.search(flair, limit = 300)` method, is used, the accuracy on the test set increases to **70%**, but if the data set is checked manually for **Title** and **Flair** pair, its found to be mislabelled. The reason for a better accuracy is that by `subreddit.search(flair, limit = 300)` method, the Flair is searched in `post.title` rather than `post.link_flair_text`, during data collection.

### References

1. [How to scrape data from Reddit using PRAW](http://www.storybench.org/how-to-scrape-reddit-with-python/)
2. [How to scrape data from Reddit using PushShift API](https://medium.com/@RareLoot/using-pushshifts-api-to-extract-reddit-submissions-fb517b286563)
3. [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
