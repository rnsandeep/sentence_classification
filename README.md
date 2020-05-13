This folder contains code for creating models both testing and training. 

The Report on the experiments is avalara_classification_submission.pdf

Exploratory Data Analysis is in two documents:
1. exploratory-title.pdf
2. exploratory-descp.pdf

All the code which was used to train the models and test is provided here.

code for lstm is provided at submission/lstm

Training.csv was splitted into ratio of (0.8, 0.2) for training and validation. 
Final best models were run on test1.csv for reporting scores.

code for word2vec is provided at submission/word2vec
     code for word2vec with svm is at submission/word2vec/svm
     code for word2vec with logistic regression is at submission/word2vec/logistic

code for tfidf is provided at submission/tfidf
     code for tfidf with svm is at submission/tfidf/svm
     code for tfidf with logistic regression is at submission/tfidf/logistic

seperate models were trained for title and description classification. 



Final prediction code using ensemble models is at submission/predict 

To replicate the ensemble score mentioned in the document please run

at submission/predict$  python3 predict_ensemble.py test1.csv 

If you want to run on different test file please change the first argument. 

Also the final submission.tsv is generated after running the above code with predictions in the last column. 
