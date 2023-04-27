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

- 03 Classification_with_a_Tabular Vector_Borne_Disease_Dataset
  ### First of all, it has to say that this contest is not a familiar classification problem. Generally, one single data in training_set is corresponds to one label, and the task is to predict the single label by using the model build with training_data and the features in testing data. However, in this contest, we need to predict the test_data for the highest 3 possibilities labels to be result by using the training data (which is organized by 1 set of data corresponding to 1 single label). So the .predict_proba() method is necessary in this task. Personally I have ignored in the first evening so my submissions score (accuracy) in the first night were no more than 0.2,lol. (And the highest score until now is about 0.44)
  - 4/26 SVC to reach 0.3852.
  - 4/27 Autogluon to reach 0.34657. Not a good one, but by using the Catboost and WeightedModelL2 which is recognized to be the best model from training procedure, the results in testing data never changed! (So I just use )
  - 4/27
  - 4/27
  
