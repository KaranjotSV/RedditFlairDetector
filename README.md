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

The following dependencies can be found in [requirements.txt](https://github.com/radonys/Reddit-Flair-Detector/blob/master/requirements.txt):

  1. [praw](https://praw.readthedocs.io/en/latest/)
  2. [scikit-learn](https://scikit-learn.org/)
  3. [nltk](https://www.nltk.org/)
  4. [pandas](https://pandas.pydata.org/)
  5. [numpy](http://www.numpy.org/)
  
### Approach

Going through various literatures available for text processing and suitable machine learning algorithms for text classification, I based my approach using [[2]](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568) which described various machine learning models like Naive-Bayes, Linear SVM and Logistic Regression for text classification with code snippets. Along with this, I tried other models like Random Forest and Multi-Layer Perceptron for the task. I have obtained test accuracies on various scenarios which can be found in the next section.

The approach taken for the task is as follows:

  1. Collect 100 India subreddit data for each of the 12 flairs using `praw` module [[1]](http://www.storybench.org/how-to-scrape-reddit-with-python/).
  2. The data includes *title, comments, body, url, author, score, id, time-created* and *number of comments*.
  3. For **comments**, only top level comments are considered in dataset and no sub-comments are present.
  4. The ***title, comments*** and ***body*** are cleaned by removing bad symbols and stopwords using `nltk`.
  5. Five types of features are considered for the the given task:
    
    a) Title
    b) Comments
    c) Urls
    d) Body
    e) Combining Title, Comments and Urls as one feature.
  6. The dataset is split into **70% train** and **30% test** data using `train-test-split` of `scikit-learn`.
  7. The dataset is then converted into a `Vector` and `TF-IDF` form.
  8. Then, the following ML algorithms (using `scikit-learn` libraries) are applied on the dataset:
    
    a) Naive-Bayes
    b) Linear Support Vector Machine
    c) Logistic Regression
    d) Random Forest
    e) MLP
   9. Training and Testing on the dataset showed the **Random Forest** showed the best testing accuracy of **77.97%** when trained on the combination of **Title + Comments + Url** feature.
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
