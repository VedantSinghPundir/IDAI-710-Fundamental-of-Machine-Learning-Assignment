''' Import Libraries'''
import pandas as pd
import numpy as np
from discriminants import GaussianDiscriminant


class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError

class Prior(Classifier):
    
    def __init__(self):
        ''' Your code here '''
        self.model_params = {}
        pass
    

    def predict(self, x):
        
        '''This method takes in x (numpy array) and returns a prediction y'''
        if "most_common_class" not in self.model_params:
            raise ValueError("Model has not been fit yet.")

        best_class = self.model_params["most_common_class"]

        # if x is a single value, return a single label
        if np.isscalar(x):
            return best_class

        # otherwise return one label per input sample
        predictions = []
        for i in range(len(x)):
            predictions.append(best_class)

        return np.array(predictions)
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x (numpy array), y (numpy array)'''
        counts = {}
        for label in y:
            if label in counts:
                counts[label] = counts[label] + 1
            else:
                counts[label] = 1

        total = len(y)

        priors = {}
        for label in counts:
            priors[label] = counts[label] / total

        most_common_class = None
        highest_prob = -1

        for label in priors:
            if priors[label] > highest_prob:
                highest_prob = priors[label]
                most_common_class = label

        self.model_params["priors"] = priors
        self.model_params["most_common_class"] = most_common_class

        return self



''' Create our Discriminant Classifier Class'''    
class DiscriminantClassifier(Classifier):
    ''''''
    def __init__(self):
        ''' Initialize Class Dictionary'''
        self.model_params = {}
        
    def set_classes(self, *discs):
        '''Pass discriminant objects and store them in self.classes
            This class is useful when you have existing discriminant objects'''
        classes = {}
        for d in discs:
            classes[d.name] = d

        self.model_params["classes"] = classes
        return self

            
    def fit(self, dataframe, label_key=['Labels'], default_disc=GaussianDiscriminant):
        ''' Calculates model parameters from a dataframe for each discriminant.
            Label_Key specifies the column that contains the class labels. ''' 
        if isinstance(label_key, list):
            label_col = label_key[0]
        else:
            label_col = label_key

        feature_cols = []
        for c in dataframe.columns:
            if c != label_col:
                feature_cols.append(c)

        labels = dataframe[label_col].unique()

        classes = {}
        counts = {}

        total_samples = len(dataframe)
        for lab in labels:
            class_df = dataframe[dataframe[label_col] == lab]
            counts[lab] = len(class_df)

            prior = counts[lab] / total_samples

            X = class_df[feature_cols].values

            if X.shape[1] == 1:
                X_use = X.flatten()
            else:
                X_use = X

            disc_obj = default_disc(data=X_use, prior=prior, name=str(lab))
            classes[str(lab)] = disc_obj

        self.model_params["classes"] = classes
        self.model_params["label_col"] = label_col
        self.model_params["feature_cols"] = feature_cols
        self.model_params["counts"] = counts

        return self
                
    
    def predict(self, x):
        ''' Returns a Key (class) that corresponds to the highest discriminant value'''
        if "classes" not in self.model_params:
            raise ValueError("You must call fit() or set_classes() before predict().")

        classes = self.model_params["classes"]

        x = np.array(x)

        if np.isscalar(x):
            x = np.array([x])

        if len(x.shape) == 1:

            best_label = None
            best_value = None

            for label in classes:
                disc = classes[label]
                value = disc.calc_discriminant(x)

                if (best_value is None) or (value > best_value):
                    best_value = value
                    best_label = label

            return best_label

        else:

            predictions = []
            for i in range(len(x)):
                sample = x[i]

                best_label = None
                best_value = None

                for label in classes:
                    disc = classes[label]
                    value = disc.calc_discriminant(sample)

                    if (best_value is None) or (value > best_value):
                        best_value = value
                        best_label = label

                predictions.append(best_label)

            return np.array(predictions)

    def pool_variances(self):
        ''' Calculates a pooled variance and sets the corresponding model params '''
        if "classes" not in self.model_params:
            raise ValueError("You must call fit() or set_classes() before pool_variances().")

        if "counts" not in self.model_params:
            raise ValueError("Counts not found. Fit the classifier using fit(dataframe, ...) first.")

        classes = self.model_params["classes"]
        counts = self.model_params["counts"]

        labels = list(classes.keys())
        K = len(labels)

        first_sigma = classes[labels[0]].params["sigma"]

        # ----- Univariate pooled variance -----
        if np.isscalar(first_sigma):
            numerator = 0
            denom = 0

            for lab in labels:
                n_i = counts[lab]
                sigma_i = classes[lab].params["sigma"]
                var_i = sigma_i * sigma_i

                numerator = numerator + (n_i - 1) * var_i
                denom = denom + (n_i - 1)

            pooled_var = numerator / denom
            pooled_sigma = np.sqrt(pooled_var)

            for lab in labels:
                classes[lab].params["sigma"] = pooled_sigma

            self.model_params["pooled_sigma"] = pooled_sigma
            return pooled_sigma

        # ----- Multivariate pooled covariance -----
        else:
            numerator = None
            denom = 0

            for lab in labels:
                n_i = counts[lab]
                sigma_i = classes[lab].params["sigma"]

                if numerator is None:
                    numerator = (n_i - 1) * sigma_i
                else:
                    numerator = numerator + (n_i - 1) * sigma_i

                denom = denom + (n_i - 1)

            pooled_sigma = numerator / denom

            for lab in labels:
                classes[lab].params["sigma"] = pooled_sigma

            self.model_params["pooled_sigma"] = pooled_sigma
            return pooled_sigma

       
        
        
        
