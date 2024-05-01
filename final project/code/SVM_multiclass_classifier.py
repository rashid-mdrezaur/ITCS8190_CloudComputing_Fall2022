# !pip install pyspark py4j

# !pip install findspark

## initialize all the libraries

import findspark
findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

import pandas as pd
import numpy as np

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import split

from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix, DenseMatrix

# initiate the spark session
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('matrix-multiplication') \
    .enableHiveSupport() \
    .getOrCreate()

## RDD multiplication using indexedMatrix for efficiency of larger dataset
def RDD_multiply(my_arr, weight):
  # print(my_arr.ndim)
  row_list = []
  if my_arr.ndim == 1:
    # print('dim: ', my_arr.ndim)
    row_list.append((0, tuple(my_arr)))
  else:
    for i in range(my_arr.shape[0]):
      tup_elem = tuple(my_arr[i])
      tup_rows = (i,tup_elem)
      row_list.append(tup_rows)
      
  indMat = IndexedRowMatrix(sc.parallelize(row_list))

  denMat = DenseMatrix(weight.shape[0], weight.shape[1], weight.T.flatten())
  mulMat = indMat.multiply(denMat)
  new_m = mulMat.toBlockMatrix().toLocalMatrix().toArray().astype('float64')

  return new_m

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] 
    return e_x / div

## calculating the multi-class SVM loss

def svm_loss(W, X, y, reg, delta=1.0):
    
    # initializing the gradient as zero
    
    dW = np.zeros(W.shape) 

    # computing the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        y_ = y[i]

        # scores = X[i].dot(W)
        scores = RDD_multiply(X[i],W)
        correct_class_score = scores[y_]
        for j in range(num_classes):
            if j == y_:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin
                dW[:,j]  += X[i]
                dW[:,y_] -= X[i]

   #average
    loss /= float(num_train)
    dW   /= float(num_train)

   # Adding a L2 regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW   += reg * W

    return loss, dW

## SVM classifier
def svmPredict(X, Y, reg_param, tts = 70):

    num_class = len(np.unique(Y))
    
    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    Xx = X.to_numpy()
    Yy = Y

    training_percentage = tts
    size_of_input_data = X.shape[0]

    Xtr = Xx[0 : (int(size_of_input_data*training_percentage/100)), :]
    Ytr = Yy[0: (int(size_of_input_data*training_percentage/100))]
    Xte = Xx[(int(size_of_input_data*training_percentage/100)) : , :]
    Yte = Yy[(int(size_of_input_data*training_percentage/100)) : ]

    W = np.zeros((Xtr.shape[1], num_class))
    ## we can introduce a loop here for epoch to get the optimum loss
    loss, dW = svm_loss(W, Xtr, Ytr, reg_param)
    # print(dW)

    # Z = -Xte@dW
    # print(Z)
    Z = RDD_multiply(Xte, dW)
    Z = -Z
    # print(Z)
    P = softmax(Z)
    pred = np.argmax(P, axis=1)
    
    acc = (pred == Yte).sum()/Yte.shape[0]
    
    return acc, dW


def main():
    ## data reading
    ## the csv formated dataset file name

    df = spark.read.csv('s3://mrashid1cloud2022/project_data/airlines_tweets_cleaned.csv', header=True)   ## use (uncomment) this line if dataset is from AWS S3 bucket
    #df = spark.read.csv('data/airlines_tweets_cleaned.csv', header=True) ##use this line if read the dataset from local directory
    df.count()
    df = df.select('text', 'target')
    df = df.dropna()
    df.count()
    # df.show(5)

    df.printSchema()
    panda_df = df.toPandas()
    sentence_list = list(panda_df['text'])
    word_lists = []
    word_token = []
    word_dict = {}
    count = 1

    for i in range(len(sentence_list)):
        my_sent = str(sentence_list[i])
        # print(type(my_sent))
        words = my_sent.split(' ')
        word_token.append(words)
        for w in words:
            if w not in word_lists:
                word_lists.append(w)
                word_dict[w]=count
                count = count+1

    input_x = np.zeros((len(panda_df), 100))

    for tok in range(len(word_token)):
        my_token = word_token[tok]

        start = (100-len(my_token))
        for i in range(len(my_token)):
            input_x[tok,start] = word_dict[my_token[i]]
            start = start+1
    new_df = pd.DataFrame(input_x)
    new_df = new_df.loc[:, (new_df != 0).any(axis=0)]
    norm_x = (new_df - new_df.min())/(new_df.max() - new_df.min())
    y_label = np.array(panda_df['target']).astype(int)
    data = norm_x.copy()
    data['target'] = y_label
    data_df = spark.createDataFrame(data)

    reg_param = .1
    acc, model_w = svmPredict(norm_x, y_label, reg_param)
    print('Multiclass SVM Testing Accuracy:', round((acc*100),2), '%')


    ## get_imp_words list
    my_w = np.abs(model_w).sum(axis=1)
    my_w = my_w[:(len(my_w)-1)]
    sorted_w = np.sort(my_w)
    top_5 = sorted_w[-3:]
    col_ind = []
    for i in top_5:
      ind = np.where(my_w == i)
      col_ind.append(ind[0][0])
    imp_col = new_df.iloc[:, col_ind]
    imp_word_index = np.array(imp_col).flatten()
    imp_word_index

    imp_word_index = imp_word_index[imp_word_index != 0]
    imp_word_index = np.unique(imp_word_index)
    num_imp_words = len(imp_word_index)
    num_total_words = len(word_dict)
    def get_key(my_dict,val):
        for key, value in my_dict.items():
            if val == value:
                return key
    imp_words_list = []
    for i in imp_word_index:
      getword = get_key(word_dict,i)
      imp_words_list.append(getword)

    with open("data/imp_words.txt", "w") as output:
        output.write(str(imp_words_list))
    print('Total number of words:', num_total_words)
    print('Important words:', num_imp_words)

if __name__ == "__main__":
    main()