# Reddit Flair Detector

Reddit Flair Detector is a web application which detects flairs of posts of subreddit india using Machine Learning algorithms. The application can be found live at [redditflairdetector-ksv](https://redditflairdetector-ksv.herokuapp.com/).

### Directory Structure

The description of files and folders can be found below:
  1. [Data-collection](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/Data-Collection) - Folder containing jupyter notebooks of data collection from r/india, contains 2 notebooks, one for each API - PRAW and PushShift.
  2. [Data](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/Data) - Folder containing CSV instances of the collected data.
  4. [Models](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/Models) - Folder containing jupyter notebooks of model training, contains 2 notebooks, each for the models fitted on the data retrieved via APIs - PRAW and PushShift .
  5. [sample](https://github.com/KaranjotSV/RedditFlairDetector/tree/master/sample) - Folder containing a sample txt file,
the txt file contains URLs of subreddit india posts, can be used for testing [/automated_testing](https://redditflairdetector-ksv.herokuapp.com/automated_testing) endpoint of the web application.
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
  
### Approach

The approach for this task is more focussed upon Data, rather than the Machine learning algorithms. Various techniques like Data Augmentation, Oversampling, Feature Extraction are applied in order to get better results.

The approach is explained in detail as follows:

  1. Collected maximum possible India subreddit data for each of the 11 flairs using `praw` module [[1]](http://www.storybench.org/how-to-scrape-reddit-with-python/). The method used for searching posts for each flair is `subreddit.search(f"flair:{flair}", limit = 300)`, another alternative is `subreddit.search(flair, limit = 300)`, but this leads to collection of mislabelled data, as it searched the flair in `post.title` rather than `post.link_flair_text`.
  2. The data included *title, comments, body, url, comments, score, id, time-created* and *number of comments*.
  3. For **comments**, only top level comments(top 10) are considered in dataset and no sub-comments are present.
  4. The ***title, comments*** and ***body*** are cleaned by removing non-english words, stopwords and bad symbols using `nltk`.
  5. Five types of features are considered for the the given task:
    
    a) Title
    b) Comments
    c) Urls
    d) Body
    e) Combining Title, Comments, Body as one feature.
    
  6. The dataset is split into **80% train** and **20% test** data using `train-test-split` of `scikit-learn`.
  7. The dataset is then converted into a `Vector` and `TF-IDF` form.
  8. The dataset is augmented to increase the number of examples, in order to avoid overfitting. Augmentation is done by shuffling the sentences and changing the words in the sentences with their synonyms, using `nltk`.
  9. Then, the following ML algorithms (using `scikit-learn` libraries) are applied on the dataset:
    
    a) Naive-Bayes
    b) Logistic Regression
    c) Random Forest
    
   9. Training and Testing on the dataset showed that **Logistic Regression** showed the best testing accuracy of **63%** when trained on the combination of **Title + Comments + Body** feature.
   10. The models were overfitting, even after tuning the hyperparameters, this led to the idea of collecting more data.
   11. 
   
   10. The best model is saved and is used for prediction of the flair from the URL of the post.
    
### Results

#### Title as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6011904762      |
| Linear SVM                 | 0.6220238095      |
| Logistic Regression        | **0.6339285714**  |
| Random Forest              | 0.6160714286      |
| MLP                        | 0.4970238095      |

#### Body as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.2083333333      |
| Linear SVM                 | 0.2470238095      |
| Logistic Regression        | 0.2619047619      |
| Random Forest              | **0.2767857143**  |
| MLP                        | 0.2113095238      |

#### URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.3005952381      |
| Linear SVM                 | **0.3898809524**  |
| Logistic Regression        | 0.3690476190      |
| Random Forest              | 0.3005952381      |
| MLP                        | 0.3214285714      |

#### Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.5357142857      |
| Linear SVM                 | 0.6190476190      |
| Logistic Regression        | **0.6220238095**  |
| Random Forest              | 0.6011904762      |
| MLP                        | 0.4761904762      |

#### Title + Comments + URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6190476190      |
| Linear SVM                 | 0.7529761905      |
| Logistic Regression        | 0.7470238095      |
| Random Forest              | **0.7797619048**  |
| MLP                        | 0.4940476190      |


### Intuition behind Combined Feature

The features independently showed a test accuracy near to **60%** with the `body` feature giving the worst accuracies during the experiments. Hence, it was excluded in the combined feature set.

### References

1. [How to scrape data from Reddit](http://www.storybench.org/how-to-scrape-reddit-with-python/)
2. [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
