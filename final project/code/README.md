## Project: Multiclass SVM classifer for sentiment analysis using spark and python

### Task: Implement a multiclass SVM classifier in a distributed fashion and compare performance with logistic regression classifier.

Raw Dataset sources: 
Airlines Tweet data: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
Sentiment140 dataset: https://www.kaggle.com/datasets/kazanova/sentiment140

----------------------------------------------------------------------------------------------------
-------------------------- **# files and folder in this zip folder#** --------------------------

1. **raw_data_cleaning.ipynb** : This scripts is used in the local machine with pyspark environment to clean and preprocess the raw data downloaded in .csv format from the above links.
2. **SVM_multiclass_classifier.ipynb** : This script is the main project script. This is the jupyter notebook file to run the project script in local machine where pyspark is installed. If not, from the commented out lines one can install the required pyspark library to create a local cluster.
3. **SVM_multiclass_classifier.py** : This script is the .py file to run the project script in AWS cluster or from the terminal/interactive console. In order to use this script, we should copy the required dataset (processed in .csv files) in the directory where this scripts is. The script also contains a line to use the dataset from the AWS S3 bucket. If onc chooses to use the S3 path file, the local data reading line should be commented out.
4. **LogReg_multiclass_classifier.ipynb**: This is the script to run the logistic regression classifier implemented from the python sklearn-libarary to compare model performance with out implemented SVM classifier.
5. **data/**: This folder contains all the imput file we need to run our scripts. If we use Airlines_tweets dataset, we need to edit the script and change the input taking command according to the dataset name.
		a. this folder has the input datasets named as **airlines_tweets_cleaned** and **sentiment140_2K.csv** to run for scripts mention in [2,3]
		b. and also datasets named as **airlines_formated_features.csv** and **sentiment_formated_features.csv** to run for script in [4]
		c. the **imp_words.txt** contains all the extracted words given from the feature importance from our SVM model

-----------------------------------**# Instructions to Run #**----------------------------------------------------------------------------------------------------------------------------------------------

If run in local machine:
Setup pyspark and install required python libraris and pyspark libaries mention in top of the scripts.
Copy the data and .ipnyb files in the same folder and run script. 

if run in cluster:
copy the .py file and the files under 'data/' folder into AWS S3 bucket and EMR cluster.
comment out the local datareading file in the 'main()' method and comment-in the AWS S3 bucket data readling line. Save the edited .py file and run on Pyspark console.