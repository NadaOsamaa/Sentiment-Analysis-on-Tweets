# Sentiment Analysis on Tweets

This project performs sentiment analysis on a dataset of tweets using machine learning algorithms. The [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) used in this project is a CSV file containing 1.6 million tweets with labels indicating whether they are positive or negative. The dataset is preprocessed and cleaned, and then split into training and testing sets. Two machine learning algorithms are implemented - Linear Support Vector Classification (Linear SVC) and Logistic Regression (LR) - to predict the sentiment of a given tweet.

## Requirements
- pandas
- numpy
- seaborn
- matplotlib
- nltk
- sklearn
- simple_colors

You can install these libraries using pip:

```
pip install pandas numpy seaborn matplotlib nltk sklearn simple_colors
```

## Dataset
The dataset used in this project is a CSV file containing 1.6 million tweets with labels indicating whether they are positive or negative.

## Data Preprocessing
The dataset is preprocessed by removing unnecessary columns, replacing the sentiment values, and performing text processing to remove symbols, lowercase the text, remove stop words, and stem the text.

## Model Fitting
Two machine learning algorithms are implemented - Linear SVC and LR - to predict the sentiment of a given tweet. The dataset is split into training and testing sets, and the algorithms are trained on the training set. The accuracy of each model is calculated, and a confusion matrix is plotted to visualize the performance of each model.
- Accuracy of SVC model was: 0.7692
- Accuracy of LR model was: 0.7778 

## Model Testing
A function is implemented to test the LR model on a given tweet. The function takes in a tweet as input and returns whether it has a positive or negative sentiment.

## Credits
- This code was created by [Nada Osama](https://github.com/NadaOsamaa)
- [Sentiment Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) was obtained from Kaggle
  
Note: The dataset contains 1.6 million records, so it may take a very long time to run. Feel free to split the data before fitting the model.
