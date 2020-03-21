#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError
import pandas as pd


class StackedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners, meta_learner=BayesianRidge(),
                 n_splits=3, shuffle=True, preprocessing=None, n_jobs=-1):
        """Uses a meta-estimator to predict from base estimators predictions
        Parameters
        ----------
        base_learners : list of sklearn Estimators
            List of base estimators to use.     
        meta_learner : sklearn Estimator
            Meta estimator to use.
        n_splits : int
            Number of cross-validation splits
        shuffle : bool
            Whether to shuffle the data
        preprocessing : sklearn Estimator
            Preprocessing pipline to apply to the data before using models
            to predict.  This saves time for heavy preprocessing workloads
            because the preprocessing does not have to be repeated for each
            estimator.
        n_jobs : int
            Number of parallel jobs to run. Default is to use as many 
            threads as there are processors.
        """
        
        # Check inputs
        if not isinstance(base_learners, list):
            raise TypeError('base_learners must be a list of estimators')
        if not isinstance(meta_learner, BaseEstimator):
            raise TypeError('meta_learner must be an sklearn estimator')
        if not isinstance(n_splits, int):
            raise TypeError('n_splits must be an int')
        if n_splits < 1:
            raise ValueError('n_splits must be positive')
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False')
        if (preprocessing is not None and
                not (hasattr(preprocessing, 'fit') and 
                     hasattr(preprocessing, 'transform'))):
            raise TypeError('preprocessing must be an sklearn transformer')
        if not isinstance(n_jobs, int):
            raise TypeError('n_jobs must be an int')
        if n_jobs is not None and (n_jobs < -1 or n_jobs == 0):
            raise ValueError('n_jobs must be None or >0 or -1')

        # Store learners as dict
        self.base_learners = dict()
        for i, learner in enumerate(base_learners):
            if (isinstance(learner, tuple) and
                    len(learner)==2 and 
                    isinstance(learner[0], str) and 
                    isinstance(learner[1], BaseEstimator)):
                self.base_learners[learner[0]] = learner[1]
            elif hasattr(learner, 'fit') and hasattr(learner, 'predict'):
                self.base_learners[str(i)] = learner
            else:
                raise TypeError('each element of base_learners must be an '
                                'sklearn estimator or a (str, sklearn '
                                'estimator) tuple')

        # Store parameters
        self.meta_learner = meta_learner
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.n_jobs = n_jobs
        
        
    def fit(self, X, y):
        """Fit the ensemble of base learners and the meta-estimator
        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable
        Returns
        -------
        self
            The fit estimator
        """

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            self.preprocessing = self.preprocessing.fit(X, y)
            Xp = self.preprocessing.transform(X)
        
        # Use base learners to cross-val predict
        preds = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        for name, learner in self.base_learners.items():
            preds[name] = cross_val_predict(learner, Xp, y, 
                                            cv=kf, n_jobs=self.n_jobs)
            
        # Fit base learners to all samples
        for _, learner in self.base_learners.items():
            learner = learner.fit(Xp, y)
            
        # Fit meta learner on base learners' predictions
        self.meta_learner = self.meta_learner.fit(preds, y)

        # Return fit object
        return self
    
                
    def predict(self, X, y=None):
        """Predict using the meta-estimator
        Parameters
        ----------
        X : pandas DataFrame
            Features
        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            Xp = self.preprocessing.transform(X)
        
        # Use base learners to predict
        preds = pd.DataFrame(index=X.index)
        for name, learner in self.base_learners.items():
            preds[name] = learner.predict(Xp)
            
        # Use meta learner to predict based on base learners' predictions
        y_pred = self.meta_learner.predict(preds)
        
        # Return meta-learner's predictions
        return y_pred


    def fit_predict(self, X, y):
        """Fit the ensemble and then predict on features in X
        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable
            
        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """
        return self.fit(X, y).predict(X)

def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error regression loss"""
    return np.sqrt(np.mean(np.square(y_true-y_pred)))

    
def cross_val_metric(model, X, y, cv=3, 
                     metric=root_mean_squared_error, 
                     train_subset=None, test_subset=None, 
                     shuffle=False, display=None):
    """Compute a cross-validated metric for a model.
    
    Parameters
    ----------
    model : sklearn estimator or callable
        Model to use for prediction.  Either an sklearn estimator 
        (e.g. a Pipeline), or a function which takes 3 arguments: 
        (X_train, y_train, X_test), and returns y_pred.
        X_train and X_test should be pandas DataFrames, and
        y_train and y_pred should be pandas Series.
    X : pandas DataFrame
        Features.
    y : pandas Series
        Target variable.
    cv : int
        Number of cross-validation folds
    metric : sklearn.metrics.Metric
        Metric to evaluate.
    train_subset : pandas Series (boolean)
        Subset of the data to train on. 
        Must be same size as y, with same index as X and y.
    test_subset : pandas Series (boolean)
        Subset of the data to test on.  
        Must be same size as y, with same index as X and y.
    shuffle : bool
        Whether to shuffle the data. Default = False
    display : None or str
        Whether to print the cross-validated metric.
        If None, doesn't print.
    
    Returns
    -------
    metrics : list
        List of metrics for each test fold (length cv)
    preds : pandas Series
        Cross-validated predictions
    """
    
    # Use all samples if not specified
    if train_subset is None:
        train_subset = y.copy()
        train_subset[:] = True
    if test_subset is None:
        test_subset = y.copy()
        test_subset[:] = True
    
    # Perform the cross-fold evaluation
    metrics = []
    TRix = y.copy()
    TEix = y.copy()
    all_preds = y.copy()
    kf = KFold(n_splits=cv, shuffle=shuffle)
    for train_ix, test_ix in kf.split(X):
        
        # Indexes for samples in training fold and train_subset
        TRix[:] = False
        TRix.iloc[train_ix] = True
        TRix = TRix & train_subset
        
        # Indexes for samples in test fold and in test_subset
        TEix[:] = False
        TEix.iloc[test_ix] = True
        TEix = TEix & test_subset
        
        # Predict using a function
        if callable(model):
            preds = model(X.loc[TRix,:], y[TRix], X.loc[TEix,:])
        else:
            model.fit(X.loc[TRix,:], y[TRix])
            preds = model.predict(X.loc[TEix,:])
        
        # Store metric for this fold
        metrics.append(metric(y[TEix], preds))

        # Store predictions for this fold
        all_preds[TEix] = preds

    # Print the metric
    metrics = np.array(metrics)
    if display is not None:
        print('Cross-validated %s: %0.3f +/- %0.3f'
              % (display, metrics.mean(), metrics.std()))
        
    # Return a list of metrics for each fold
    return metrics, all_preds


""" Dexième implémentation de méthode de stacking. """

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        if clf != SVR and clf != KNeighborsRegressor:
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


""" Simpler implementation but harder to change (longer). This one is pure imperativ based on blending and not on Kfold. """

def stacking(train_x, train_y, test_x):
    m = int(train_x.shape[0]/4)
    train_data_split_1 = train_x.iloc[:m,:]
    train_data_split_2 = train_x.iloc[m+m:m+m+m,:]
    train_data_split_3 = train_x.iloc[m:m+m,:]
    train_data_split_4 = train_x.iloc[m+m+m : , :]
    train_labels_split_1, train_labels_split_2 = train_y.iloc[:m], train_y.iloc[m+m:m+m+m]
    train_labels_split_3, train_labels_split_4 =  train_y.iloc[m:m+m], train_y.iloc[m+m+m :]
    
    rf_1 = RandomForestRegressor(bootstrap = True,
                           max_depth = 40,
                           max_features = 'auto',
                           min_samples_leaf = 1,
                           min_samples_split = 2,
                           n_estimators = 102)
    
    tree_1 = DecisionTreeRegressor(criterion='friedman_mse', 
                                 max_depth=41,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_samples_leaf=20, 
                                 min_samples_split=10,
                                 splitter='random')

    
    rf_2 = RandomForestRegressor(n_jobs = -1, 
                                 n_estimators = 56,
                                 min_samples_split = 2,
                                 min_samples_leaf = 1,
                                 max_features = 'auto',
                                 max_depth = 50,
                                 bootstrap = True,
                                 verbose = 0)
    """rf_3 = RandomForestRegressor(n_jobs = -1, 
                                 n_estimators = 100,
                                 min_samples_split = 5,
                                 min_samples_leaf = 1,
                                 max_features = 'auto',
                                 max_depth = 30,
                                 bootstrap = True,
                                 verbose = 0)"""
    
    #svr_1 = SVR(C=200,epsilon=0.8)
    #svr_2 = SVR(C=400,epsilon=0.9)
    #knn_1 = KNeighborsRegressor(n_neighbors=20)
    knn_2 = BayesianRidge()
                          
    tree_2 = DecisionTreeRegressor(criterion='mse', 
                                   max_depth=30,
                                   max_features='auto',
                                   max_leaf_nodes=None,
                                   min_samples_leaf=20, 
                                   min_samples_split=10,
                                   splitter='random')
    
    ### layer 1
    
    rf_1.fit(train_data_split_1,train_labels_split_1)
    rf_2.fit(train_data_split_1,train_labels_split_1)
    #rf_3.fit(train_data_split_1,train_labels_split_1)
    tree_1.fit(train_data_split_1,train_labels_split_1)
    tree_2.fit(train_data_split_1,train_labels_split_1)
    #svr_1.fit(train_data_split_1,train_labels_split_1)
    #svr_2.fit(train_data_split_1,train_labels_split_1)
    #knn_1.fit(train_data_split_1,train_labels_split_1)
    knn_2.fit(train_data_split_1,train_labels_split_1)

    print('learned first layer')
    
    prediction_rf_1 = rf_1.predict(train_data_split_2)
    prediction_rf_2 = rf_2.predict(train_data_split_2)
    #prediction_rf_3 = rf_3.predict(train_data_split_2)
    prediction_tree_1 = tree_1.predict(train_data_split_2)
    prediction_tree_2 = tree_2.predict(train_data_split_2)
    #prediction_svr_1 = svr_1.predict(train_data_split_2)
    #prediction_svr_2 = svr_2.predict(train_data_split_2)
    #prediction_knn_1 = knn_1.predict(train_data_split_2)
    prediction_knn_2 = knn_2.predict(train_data_split_2)
    
    input_stack = pd.DataFrame(list(zip(prediction_rf_1,prediction_rf_2,
                                        prediction_tree_1,prediction_tree_2,prediction_knn_2)), 
                               columns =['rf_1', 'rf_2','tree_1','tree_2','knn_2']) 
    
    rf_stack = RandomForestRegressor(bootstrap = True,
                           max_depth = 40,
                           max_features = 'auto',
                           min_samples_leaf = 1,
                           min_samples_split = 2,
                           n_estimators = 102)
    
    tree_stack = DecisionTreeRegressor(criterion='friedman_mse', 
                                 max_depth=41,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_samples_leaf=20, 
                                 min_samples_split=10,
                                 splitter='random')
    
    knn_stack = BayesianRidge()
    
    print('predict on first layer')
    
    ### layer 2 training
    
    rf_stack.fit(input_stack,train_labels_split_2)
    tree_stack.fit(input_stack,train_labels_split_2)
    knn_stack.fit(input_stack,train_labels_split_2)
    
    print('learned second layer')
    ### layer 2
    
    prediction_rf_1_1 = rf_1.predict(train_data_split_3)
    prediction_rf_2_1 = rf_2.predict(train_data_split_3)
    #prediction_rf_3_1 = rf_3.predict(train_data_split_3)
    prediction_tree_1_1 = tree_1.predict(train_data_split_3)
    prediction_tree_2_1 = tree_2.predict(train_data_split_3)
    #prediction_svr_1_1 = svr_1.predict(train_data_split_3)
    #prediction_svr_2_1 = svr_2.predict(train_data_split_3)
    #prediction_knn_1_1 = knn_1.predict(train_data_split_3)
    prediction_knn_2_1 = knn_2.predict(train_data_split_3)
    
    input_stack = pd.DataFrame(list(zip(prediction_rf_1_1,prediction_rf_2_1,
                                        prediction_tree_1_1,prediction_tree_2_1,prediction_knn_2_1)), 
                               columns =['rf_1', 'rf_2','tree_1','tree_2','knn_2']) 
    
    
    prediction_rf_stack = rf_stack.predict(input_stack)
    prediction_tree_stack = tree_stack.predict(input_stack)
    prediction_knn_stack = knn_stack.predict(input_stack)
    
    input_stack_2 = pd.DataFrame(list(zip(prediction_rf_stack,prediction_tree_stack,prediction_knn_stack)),
                                 columns = ['rf_stack','tree_stack','knn_stack'])
    
    ### layer 3 training
    
    rf_stack_2 = RandomForestRegressor(bootstrap = True,
                           max_depth = 40,
                           max_features = 'auto',
                           min_samples_leaf = 1,
                           min_samples_split = 2,
                           n_estimators = 102)
    
    tree_stack_2 = DecisionTreeRegressor(criterion='friedman_mse', 
                                 max_depth=41,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_samples_leaf=20, 
                                 min_samples_split=10,
                                 splitter='random')
    
    knn_stack_2 = BayesianRidge()
    
    
    print('predict second layer')
    
    rf_stack_2.fit(input_stack_2,train_labels_split_3)
    tree_stack_2.fit(input_stack_2,train_labels_split_3)
    knn_stack_2.fit(input_stack_2,train_labels_split_3)
    
    ### layer 3
    
    print('learned third layer')

    prediction_rf_1_1 = rf_1.predict(train_data_split_4)
    prediction_rf_2_1 = rf_2.predict(train_data_split_4)
    #prediction_rf_3_1 = rf_3.predict(train_data_split_4)
    prediction_tree_1_1 = tree_1.predict(train_data_split_4)
    prediction_tree_2_1 = tree_2.predict(train_data_split_4)
    #prediction_svr_1_1 = svr_1.predict(train_data_split_4)
    #prediction_svr_2_1 = svr_2.predict(train_data_split_4)
    #prediction_knn_1_1 = knn_1.predict(train_data_split_4)
    prediction_knn_2_1 = knn_2.predict(train_data_split_4)
    
    input_stack = pd.DataFrame(list(zip(prediction_rf_1_1,prediction_rf_2_1,
                                        prediction_tree_1_1,prediction_tree_2_1,prediction_knn_2_1)), 
                               columns =['rf_1', 'rf_2','tree_1','tree_2','knn_2']) 
    
    
    prediction_rf_stack = rf_stack.predict(input_stack)
    prediction_tree_stack = tree_stack.predict(input_stack)
    prediction_knn_stack = knn_stack.predict(input_stack)
    
    input_stack_2 = pd.DataFrame(list(zip(prediction_rf_stack,prediction_tree_stack,prediction_knn_stack)),
                                 columns = ['rf_stack','tree_stack','knn_stack'])
    
    
    prediction_rf_stack_2 = rf_stack_2.predict(input_stack_2)
    prediction_tree_stack_2 = tree_stack_2.predict(input_stack_2)
    prediction_knn_stack_2 = knn_stack_2.predict(input_stack_2)
    
    print('predict third layer')
    
    ### layer 4
    
    input_blending = pd.DataFrame(list(zip(prediction_rf_stack_2,prediction_tree_stack_2,prediction_knn_stack_2)),
                                 columns = ['rf_stack','tree_stack','knn_stack'])
    
    
    br_blending = BayesianRidge()
    br_blending.fit(input_blending,train_labels_split_4)
    
    print('learned blender')

    ### test 
    
    test_rf_1 = rf_1.predict(test_x)
    test_rf_2 = rf_2.predict(test_x)
    #test_rf_3 = rf_3.predict(test_x)
    test_tree_1 = tree_1.predict(test_x)
    test_tree_2 = tree_2.predict(test_x)
    #test_svr_1 = svr_1.predict(test_x)
    #test_svr_2 = svr_2.predict(test_x)
    #test_knn_1 = knn_1.predict(test_x)
    test_knn_2 = knn_2.predict(test_x)
    
  
    test_stack = pd.DataFrame(list(zip(test_rf_1,test_rf_2,
                                        test_tree_1,test_tree_2,test_knn_2)), 
                               columns =['rf_1', 'rf_2','tree_1','tree_2','knn_2'])
    
    
    test_rf_stack = rf_stack.predict(test_stack)
    test_tree_stack = tree_stack.predict(test_stack)
    test_knn_stack = knn_stack.predict(test_stack)
    
    
    test_stack_2= pd.DataFrame(list(zip(test_rf_stack,test_tree_stack,test_knn_stack)),
                                 columns = ['rf_stack','tree_stack','knn_stack'])
    
    test_rf_stack_2 = rf_stack_2.predict(test_stack_2)
    test_tree_stack_2 = tree_stack_2.predict(test_stack_2)
    test_knn_stack_2 = knn_stack_2.predict(test_stack_2)
    
    
    test_blending= pd.DataFrame(list(zip(test_rf_stack_2,test_tree_stack_2,test_knn_stack_2)),
                                 columns = ['rf_stack','tree_stack','knn_stack'])
    
    predictions_test = br_blending.predict(test_blending)
    
    return(predictions_test)
