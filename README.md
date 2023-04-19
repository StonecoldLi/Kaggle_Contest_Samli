# Kaggle_Contest_Samli

### My code for kaggle contest
- 01 What's cooking
  - 4/3 with CountVectorizer and LogisticRegression by tuning params with GridSearchCV
  - 4/5 using Ensembling Model & model stacking technique(failed to smoothly tune the parameters)
  - 4/7 using Ensembling Model (LogisticRegression, SVC, RandomForest) to achieve the accuracy of 0.79002
  - 4/8 using autogluon to achieve the accuracy of 0.80108

- 02 House Prices - Advanced Regression Techniques
  - 4/10 Create a Nerual Network (Linear funtion with just 1 Linear layer or MLP, but the latter one may easily cause overfitting problem) and tried a bit combination of the learning_rate, weight_decay and other params. Reaching a LMSE about 0.14851. （The structure in this model are taught by Mu Li）
  - 4/15 Tried autogluon and get a LMSE of 0.12644 
  - 4/18 Do some preprocessing to the data using pandas (especially:Remove the data which contains too much NaNs, OrdinalEncoding the columns which the values of them do have levels (like in this case : excellent, good, average......), and OneHotEncoding other normal object columns). Using model is the ensemble_model ensembled randomforestregressor and sgdregressor. Actually, tree-based-ensemble model like GradientBoostingRegressor did pretty well on the training_set but it turn out to overfitting the data since the test_loss are nearly 3% higher than the train_loss. All in all, this method finally got a score (LMSE) about 0.133. Not bad at all.
  - 4/19 The only main difference between this model and 4.18 ones is I only use sklearn to create a workflow, without pandas. And I just simply use OnehotEncoder to the object data (Seems do better than trying OridnalEncoder to some of them, Lmao). Get a score of LMSE about 0.1287. Pretty decent! 
