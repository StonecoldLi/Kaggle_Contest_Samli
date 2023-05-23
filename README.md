# Kaggle_Contest_Samli

## My code for kaggle contest
### 01 What's cooking
  - 4/3 with CountVectorizer and LogisticRegression by tuning params with GridSearchCV
  - 4/5 using Ensembling Model & model stacking technique(failed to smoothly tune the parameters)
  - 4/7 using Ensembling Model (LogisticRegression, SVC, RandomForest) to achieve the accuracy of 0.79002
  - 4/8 using autogluon to achieve the accuracy of 0.80108

### 02 House Prices - Advanced Regression Techniques
  - 4/10 Create a Nerual Network (Linear funtion with just 1 Linear layer or MLP, but the latter one may easily cause overfitting problem) and tried a bit combination of the learning_rate, weight_decay and other params. Reaching a LMSE about 0.14851. （The structure in this model are taught by Mu Li）
  - 4/15 Tried autogluon and get a LMSE of 0.12644 
  - 4/18 Do some preprocessing to the data using pandas (especially:Remove the data which contains too much NaNs, OrdinalEncoding the columns which the values of them do have levels (like in this case : excellent, good, average......), and OneHotEncoding other normal object columns). Using model is the ensemble_model ensembled randomforestregressor and sgdregressor. Actually, tree-based-ensemble model like GradientBoostingRegressor did pretty well on the training_set but it turn out to overfitting the data since the test_loss are nearly 3% higher than the train_loss. All in all, this method finally got a score (LMSE) about 0.133. Not bad at all.
  - 4/19 The only main difference between this model and 4.18 ones is I only use sklearn to create a workflow, without pandas. And I just simply use OnehotEncoder to the object data (Seems do better than trying OridnalEncoder to some of them, Lmao). Get a score of LMSE about 0.1287. Pretty decent! 

### 03 Classification_with_a_Tabular Vector_Borne_Disease_Dataset

  - First of all, it has to say that this contest is not a familiar classification problem. Generally, one single data in training_set is corresponds to one label, and the task is to predict the single label by using the model build with training_data and the features in testing data. However, in this contest, we need to predict the test_data for the highest 3 possibilities labels to be result by using the training data (which is organized by 1 set of data corresponding to 1 single label). So the .predict_proba() method is necessary in this task. Personally I have ignored in the first evening so my submissions score (accuracy) in the first night were no more than 0.2,lol. (And the highest score until now is about 0.44) Since I am not a doctor, I did not do much feature engineering process in this contest.

  - 4/26 SVC to reach 0.3852.
  - 4/27 Autogluon to reach 0.34657. Not a good one, but by using the Catboost which seems to be recognized as the best model (with WeightedEnsemble_L2) from training procedure, the prediction probability has never changed in testing data! (So I just use WeightedEnsemble_L2)
  - 4/27 Model_Ensembling of SVC and Bernoulli bayes function. 0.39183 
  - 4/27 GridSearchCV of Bernoulli bayes (Since Bernoulli is good at dealing with the discrete data) function. 0.39955
  - 4/28 GridSearchCV(2.0) of Bernoulli bayes ----> 0.40176
  - 4/29 Do some feature Engineering (combined several columns) to reach the scores of 0.40286. 【I've tried votingclassifier to ensemble model, while it did improve the accuracy score when doing cross_validation_test on training data, the submission score of this function did not reach 0.4 somehow】
  #### Private Ranking: 353/934

### 04 Classify leaves
  - All the progress of this task is based on Google Colab (without changing the existed tuned-params of each pretained-Net)
  - 4/28 train resnet34_pretrained model and did not tune the params except the Linear layer ----> 0.79022
  - 4/28 train densenet_pretrained model ----> 0.77977
  - 4/29 train vgg_11_pretrained model ----> 0.71931
  
### 05 Prediction of Wild Blueberry Yield (MAE score, which is the lower the better)
  - 5/1 EDA.ipynb (Exploratory Data Analysis) gives a quick glance of the data correlation. Like which columns are highly correlated by using .corr()
  - 5/1 Try Autogluon to set the baseline of AutoML ----> As a regression problem, MAE score:358.21564 【autogluon_ver1.0.ipynb】; As a multiclass(Classification) problem, since the target predicting column (yield) only have 776 different values compared with totally 15289 rows of data, MAE score: 416.21691【autogluon_ver1.1.ipynb】
  - 5/3 Try Model Ensembing gradient_bosting_regressor and hist_gradient_bosting_regressor, MAE score: 353.13531. Finally beat the provided baseline of this contest.【Model_selection.ipynb】
  - 5/3 Try Model Ensembing with catBoostRegressor (without high-correlated feature deletion) and Lgbm(with high-correlated feature deletion), MAE score: 342.83692. 【Catboost_lgb.ipynb】(If we simply use votingregressor provided by sklearn and do not delete any features , we would get a MAE score of 342.92801)
   - 5/3 Try Model Ensembing with catBoostRegressor (without high-correlated feature deletion) ,Lgbm(with high-correlated feature deletion) and HistGradientBoostingRegressor, MAE score: 341.84440. 【Catboost_lgb_hgbr.ipynb】
   - 5/6 Try tuning the params of lgbm and get a Mae score of 341.48407.【Catboost_lgb_hgbr_tuning_params.ipynb】
   - 5/7 Try a little trick during predicting the results and get 341.32303.【Catboost_lgb_hgbr_tuning_params_tricks.ipynb】
  #### Private Ranking: 411/1875

### 06 Prediction of Wild Blueberry Yield (RMSE score, lower the better)
  - 5/16 Created the Baseline by using RandomForestRegressor 【RMSE: 0.083981】
  - 5/18 Created the VotingRegressor with the extra data and imputed the missing values by using SimpleImputer (startegy='median') with 5 Regressor Models (GradientBoostingRegressor, HistGradientBoostingRegressor, Catboost, LGBM, XGB) 【RMSE: 0.076014】
  - 5/20 Created the VotingRegressor with the extra data but care nothing on NaNs. Since that, we need Regressor Models which can handling the missing values by themselves to do the VotingRegressor. (More to see: https://www.kaggle.com/competitions/playground-series-s3e15/discussion/411353) 【RMSE: 0.075537】
  - 5/21 Use KNNImputer (default k_neighbors=20) as the imputation method for the combined_training_data. After check each regression method, found to combine 3 of the regression method ca reach the best score.(see details in the .ipynb file)【RMSE: 0.075316】
  - 5/21 Based on the above model (after established the VotingRegressor), tuning the k_neighbors in KNNImputer and get a better score. 【RMSE: 0.07526】

### 07 Natural Language Processing with Disaster Tweets
  - 5/10 Use CountVectorizer & BernoulliBayes to get a point of 0.79436
  - 5/22 Use pretrained BERT (ver2:cased;ver3:uncased) to get 0.81979(highest one) and 0.81673 (for ver3) code reference: leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
  - 5/23 Checking the transformer (BERT) "official" reference on huggingface (https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/bert#transformers.BertForSequenceClassification) and get a score of 0.82623 (the training process is weird orz)
