import pandas as pd
from sklearn import preprocessing
import numpy as np
import random
from sklearn.base import TransformerMixin
from sklearn import base

class ComputeNANPriceBasedOnCabinsFirstLetter(TransformerMixin):

    def __init__(self,cabin,fare):
        self.cabin =cabin
        self.fare=fare
        pass

    def fit(self, X):
        return self

    def func(self,val):
        if pd.isnull(val[self.fare]):
            cabinclass = val['cabin_stripped']
            if pd.notnull(cabinclass):
                val[self.fare]= self.stats.loc[cabinclass][0]
        return val[self.fare]

    def transform(self, X):
        # strip everything from cabin number (column 'D') except the first letter
        cabin_stripped = X[self.cabin].str.strip(' ').str[0]

        #now set all the NaNs to 'Z'
        cabin_stripped[pd.isnull(cabin_stripped)] = 'Z'

        #convert categorical to int
        le = preprocessing.LabelEncoder()
        le.fit(cabin_stripped)
        cabin_stripped = le.transform(cabin_stripped)

        #add as a temp column to X
        X =X.assign(cabin_stripped = cabin_stripped)

        # now make a list of the mean for each class of cabin
        self.stats = X.groupby('cabin_stripped').agg({self.fare: [np.mean]})

        #use those mean values and cabin letters to get NaN fares
        X[self.fare] = X.apply(self.func, axis = 1)

          #dump the tmp column
        # if 'cabin_stripped' in X.columns:
        #     X.drop('cabin_stripped',axis=1,inplace = True)

        return X


class ComputeNaNCabinsFirstLetter_basedon_Fare(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,cabin,fare, strip_all_but_first_letter_of_cabin = True):
        self.cabin =cabin
        self.fare=fare
        self.strip_all_but_first_letter_of_cabin = strip_all_but_first_letter_of_cabin
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def copy_stripped(self,val):
        '''
        if 'cabin_stripped' is NaN then leave the existing singe char value in self.cabin
        otherwise copy in the single value from cabin_stripped
        :param val:
        :return:
        '''
        if pd.isnull(val['cabin_stripped']):
            return val[self.cabin]
        else:
            return val['cabin_stripped']


    def func(self,val):
        '''
        for every NaN cabin value;
        take its price, find the closest average price in self.stats.mean
        find the index associated with that column
        get the cabin letter
        assign that cabin letter to the missing cabin number
        :param val:
        :return:
        '''
        if pd.isnull(val[self.cabin])and pd.notnull(val[self.fare]):
            #the trick subtract
            tmp = (self.stats.loc[:,self.fare]-val[self.fare]).abs().sort_values(by = ['mean'])[:1]
            val[self.cabin] = tmp.index[0]
            return val[self.cabin]
        else:
            return val[self.cabin]

    def transform(self, X,**transform_params):
        # strip everything from cabin number except the first letter
        cabin_stripped = X[self.cabin].str.strip(' ').str[0]

        # add as a temp column to X
        X = X.assign(cabin_stripped=cabin_stripped)

        # now make a list of the mean for each class of cabin
        self.stats = X.groupby('cabin_stripped').agg({self.fare: [np.mean]})

        #now for each nan lets find the closest mean and assign that cabin letter
        X[self.cabin] =  X.apply(self.func,axis=1) #apply to each row

        if (self.strip_all_but_first_letter_of_cabin == True):
            X[self.cabin]  = X.apply(self.copy_stripped, axis=1)

        #remove temp column
        X= X.drop('cabin_stripped', axis=1)

        return X

class AgeImputer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,sex,age):
        self.age = age
        self.sex = sex

    def avg(self,val):
        if pd.isnull(val[self.age]):
            sex = val['Sex']
            if pd.notnull(sex):
                val[self.age] = self.stats.loc[sex][0]
        return val[self.age]


    def transform(self, df, **transform_params):
        # now make a list of the mean for each class of cabin
        self.stats = df.groupby(self.sex).agg({self.age: [np.mean]})
        df[self.age] = df.apply(self.avg, axis=1)

        return df

    def fit(self, df, y=None, **fit_params):
        return self


class GetDummiesCatCols(base.BaseEstimator, base.TransformerMixin):
    """Replace `cols` with their dummies (One Hot Encoding).
    `cols` should be a list of column names holding categorical data.
    Furthermore, this class streamlines the implementation of one hot encoding
    as available on [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """

    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df, **transform_params):
        cols_dummy = pd.get_dummies(df[self.cols])
        df = df.drop(self.cols, axis=1)
        df = pd.concat([df, cols_dummy], axis=1)
        return df

    def fit(self, df, y=None, **fit_params):
        return self

class FinalEstimatorThatdoesNothing(base.BaseEstimator, base.TransformerMixin):
    """For a pipeline, all intermediate estimators have fit and transform called
     the finalestimator has just fit called.  This is the finaldo nothing estimator"""
    def transform(self, df, **transform_params):
        return df
    def fit(self, df, y=None, **fit_params):
        return df

def main():
    data = [
        ['male', 10.0,     'a', 1, 'A27',      95 ],
        ['female', np.nan,      'b', 1, 'A761',     110.1],
        ['male', 20,       'b', 1, 'B12',      200.0],
        ['female', 100,      'b', 1,  np.nan,    475],
        ['male', 10,       'b', 1, 'B72',      315],
        ['male', np.nan,   'b', 2, 'C121',     25],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ]
    df = pd.DataFrame(data, columns=['Sex', 'Age', 'A', 'B','Cabin','Fare'])
    # direct transform test
    # df = AgeImputer('Sex', 'Age').transform(df)
    # df = ComputeNaNCabinsFirstLetter_basedon_Fare('Cabin', 'Fare').transform(df)
    # df = ComputeNaNCabinsFirstLetter_basedon_Fare('Cabin', 'Fare', strip_all_but_first_letter_of_cabin=True).transform(
    #     df)
    # df = GetDummiesCatCols(cols=['Sex','A', 'Cabin']).transform(df)
    # X = ComputeNANPriceBasedOnCabinsFirstLetter('Cabin', 'Fare').transform(X)


    #now lets run them in a pipeline
    from sklearn.pipeline import Pipeline
    ppl = Pipeline([
        ('AgeImputer', AgeImputer('Sex', 'Age')),
        ('CNaN_TO_Cabin_First_letter', ComputeNaNCabinsFirstLetter_basedon_Fare('Cabin', 'Fare',strip_all_but_first_letter_of_cabin = False)),
        # ('onehot_sex', GetDummiesCatCols(cols=['Sex', 'Cabin'])),
        ('onehot_sex', GetDummiesCatCols(cols=['Sex'])),
        ('FinalEstimatorThatdoesNothing', FinalEstimatorThatdoesNothing())
    ])

    # will call transform method of all estimators, make sure they take the
    #same datastructure as they return
    print("number of training rows: " + str(df.shape[0]))
    df_new = ppl.transform(df)
    print("number of training rows: " + str(df.shape[0]))

    #will call fit THEN TRANSFORM methods for every estimator, return type is a pipeline
    ppl = ppl.fit(df)
    pass

if __name__ == '__main__':
    main()
