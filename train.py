# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from text_transformers import ComputeNANPriceBasedOnCabinsFirstLetter
from text_transformers import ComputeNaNCabinsFirstLetter_basedon_Fare
from text_transformers import GetDummiesCatCols
from text_transformers import AgeImputer

# Load the data
train_df = pd.read_csv('./input/train.csv', header=0)
test_df = pd.read_csv('./input/test.csv', header=0)

#how many rows?
print ("number of training rows: " + str(train_df.shape[0]))
print("number nulls: "+ str(train_df.isnull().sum()))
# print ("number unique cabins:" + str(train_df.Cabin.unique))

# I think the idea here is that people with recorded cabin numbers
# are of higher socioeconomic class, and thus more likely to survive. Thanks for the tips, @salvus82 and Daniel Ellis!
# train_df["CabinBool"] = (train_df["Cabin"].notnull().astype('int'))
# test_df["CabinBool"] = (test_df["Cabin"].notnull().astype('int'))

#lets create a pipeline to transform the data
from sklearn.pipeline import Pipeline
ppl = Pipeline([
        ('AgeImputer', AgeImputer('Sex', 'Age')),
        ('CNaN_TO_Cabin_First_letter', ComputeNaNCabinsFirstLetter_basedon_Fare('Cabin', 'Fare',strip_all_but_first_letter_of_cabin = True)),
        ('onehot_sex', GetDummiesCatCols(cols=['Sex', 'Cabin','Embarked' ]))
        # ('onehot_sex', GetDummiesCatCols(cols=['Sex', 'Embarked' ]))
    ])
# run transform method on all estimators
train_df = ppl.transform(train_df)
test_df = ppl.transform(test_df)

# get rid of these useless columns
del train_df['Name']
del train_df['Ticket']

del test_df['Name']
del test_df['Ticket']

#>70% of cabins missingso drop cabin column
# del test_df['Cabin']
# del train_df['Cabin']

print ("number of training rows: " + str(train_df.shape[0]))
print("number nulls: "+ str(train_df.isnull().sum()))
# print ("number unique cabins:" + str(train_df.Cabin.unique))

#have many NaN cabins, lets pull cabin and fare and plot to see relationship
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,6))
# plt.scatter(train_df['Fare'], train_df['Cabin'])
# plt.xlabel('index', fontsize=12)
# plt.ylabel('logerror', fontsize=12)
# plt.show()


train_y = train_df['Survived']
del train_df['Survived']

#add so we have same number of features in train and test
test_df.insert(15,'Cabin_T',pd.Series(np.zeros(test_df.shape[0])))

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=3000, learning_rate=0.05).fit(train_df, train_y)
predictions = gbm.predict(test_df)


#lets add the predictions to the test data, this gives us more (but not as accurate data)
#this is not cheating since I did not use the correct Survived values on my dataset
#just the ones I predicted
test_df_trn = test_df

merged = [train_df,test_df_trn]
train_all = pd.concat(merged)
print('train_all length'+ str(len(train_all)))

merged_y = [train_y , pd.Series(predictions)]
train_all_y = pd.concat(merged_y)
print('train_all_y length'+ str(len(train_all_y)))

# run it through XGBoost again
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=3000, learning_rate=0.05).fit(train_all, train_all_y)
predictions = gbm.predict(test_df)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

