import itertools
import numpy as np
import pandas as pd
from math import log, sqrt
from scipy.stats import skew
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import threading

def remove_outliers(ds, outliers):
    for outlier in outliers:
        ds = ds[ds.index != outlier]
    return ds

def rmsd(ys, predictions):
    if type(ys) == pd.core.series.Series:
        ys = ys.as_matrix()
    sum_ = 0.0
    for idx, pred in enumerate(predictions):
        delta = 0.0
        if type(pred) == list: pred = pred[0]
        if pred > 0.0:
            delta = pred.copy()
        delta -=  ys[idx]
        sum_ += np.power(delta, 2)
    return sqrt(sum_/predictions.shape[0])

def rmsdl(ys, predictions):
    if predictions[predictions<=0].shape[0] != 0:
        return float('inf')
    return rmsd(np.log(ys), np.log(predictions))

def rmsd_sqrt(ys, predictions):
    return rmsdl(ys ** 2, predictions ** 2)

def rmsd_inv(ys, predictions):
    return rmsdl(1000.0/ys, 1000.0/predictions)

def rmsd_div1000(ys, predictions):
    return rmsdl(1000.0*ys, 1000.0*predictions)

def xg_eval_rmsd(yhat, dtrain):
    y = dtrain.get_label()
    return 'rmsd', rmsd(y, yhat)

def xg_eval_rmsdl(yhat, dtrain):
    y = dtrain.get_label()
    return 'rmsdl', rmsdl(y, yhat)

def write_results(ids, preds, score, name, output_dir):
    result = pd.DataFrame(preds, columns=['SalePrice'])
    result["Id"] = ids
    result = result.set_index("Id")

    now = datetime.now()
    sub_file = output_dir + '/submission_%s_%0.4f_%s.csv' % (name,
        score, str(now.strftime("%Y-%m-%d-%H-%M")))
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='Id')

def get_mean_delta(ds):
    return np.mean(ds,axis=0), (np.max(ds,axis=0) - np.min(ds,axis=0))

def feature_normalize(ds, mean, delta):
    return (ds - mean)/delta

# Re-order the categorical features in the training data
def get_feature_order(ds, feature):
    medians = ds.groupby([feature])['SalePrice'].median()
    return sorted(medians.keys(), key=lambda x: medians[x])

def encode_by_SalePrice(ds_old, ds_with_price):
    ds = ds_old.copy()
    for feature in ds.columns.values:
            ftype = ds[feature].dtype
            if ftype == np.float64 or ftype == np.float32 or \
                    ftype == np.int64 or ftype == np.int32:
                    continue
            order = get_feature_order(ds_with_price, feature)
            class_mapping = {v: order.index(v) for v in order}
            ds[feature] = ds[feature].map(class_mapping)
    assert(ds.isnull().sum().sum() == 0)
    return ds

def encode_with_dummies(ds):
    return pd.get_dummies(ds)

def add_sold_age(ds):
    # Transform YrSold and MoSold by the number of months till 2016/01
    age_months = (2016 - ds.YrSold) * 12 - ds.MoSold + 1
    new_ds = ds.drop(['MoSold', 'YrSold'], axis=1).copy()
    new_ds['SoldMonths'] = age_months
    return new_ds

def log_skew_features(ds):
    # compute skewness
    skewed_feats = ds.apply(lambda x: skew(x))
    
    # filter skewed features
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    
    # log transform skewed features
    ds_skew = ds.copy()
    ds_skew[skewed_feats.index] = np.log1p(ds_skew[skewed_feats.index])
    
    # check we didn't add NaNs
    assert(ds_skew.isnull().sum().sum() == 0)
    return ds_skew

def skf_reg_split(skf, X, Y):
    n_folds = skf.get_n_splits()
    labels = np.arange(0, n_folds, 1)
    Y_binned = pd.cut(Y, n_folds, labels=labels)
    return skf.split(X, Y_binned)

def transform(original, transformation):
    if transformation is None:
        return original
    elif transformation == 'sqrt':
        return np.sqrt(original)
    elif transformation == 'log':
        return np.log(original)
    elif transformation == 'inv':
        return 1000.0 / original
    elif transformation == 'div1000':
        return original / 1000.0
    raise 'transform: Unknow transformation %s' % str(transformation)


def untransform(original, transformation):
    if transformation is None:
        return original
    elif transformation == 'sqrt':
        return original ** 2
    elif transformation == 'log':
        return np.exp(original)
    elif transformation == 'inv':
        return 1000.0 / original
    elif transformation == 'div1000':
        return original * 1000.0
    raise 'untransform: Unknow transformation %s' % str(transformation)

def train_model(model_name, clf, X, Y_orig, S, train_preds, submit_preds,
            n_folds, min_score, thread_lock, random_state,
            fit_with_val=False, scores_dict=None, transformation=None):
    Y = transform(Y_orig, transformation)
    thread_lock.acquire()
    train_preds[model_name] = np.zeros([X.shape[0],])
    submit_preds[model_name] = np.zeros([S.shape[0],])
    thread_lock.release()
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
    for train_index, test_index in skf_reg_split(skf, X, Y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]
        if fit_with_val:
            clf.fit(X_train, Y_train, X_val, Y_val)
        else:
            clf.fit(X_train, Y_train)
        pred_val = clf.predict(X_val)
        pred_submit = clf.predict(S) / n_folds
        thread_lock.acquire()
        train_preds[model_name].iloc[test_index] = pred_val
        submit_preds[model_name] += pred_submit
        thread_lock.release()
    train_preds[model_name] = untransform(train_preds[model_name], 
        transformation)
    submit_preds[model_name] = untransform(submit_preds[model_name],
        transformation)
    score = rmsdl(Y_orig, train_preds[model_name])
    if scores_dict is not None:
        scores_dict[model_name] = score
    if score > min_score:
        thread_lock.acquire()
        train_preds.pop(model_name)
        submit_preds.pop(model_name)
        thread_lock.release()
        #print('%s discarded!!!: RMSD score on training data: %0.4f' % (
        #    model_name, score))
        #print('%s: Terminates training %s' % (str(threading.currentThread().getName()),
        #    model_name))
        return score
    
    #print('%s: RMSD score on training data: %0.4f' % (model_name,
    #        score))
    #print('%s: Terminates training %s' % (str(threading.currentThread().getName()),
    #    model_name))
    return score

def train_model_feature(model_name, clf, X, Y_orig, S, feature,
        train_preds, submit_preds, n_folds, min_score, thread_lock,
        random_state, fit_with_val=False, scores_dict=None,
        transformation=None):
    Y = transform(Y_orig, transformation)
    thread_lock.acquire()
    train_preds[model_name] = np.zeros([X.shape[0],])
    submit_preds[model_name] = np.zeros([S.shape[0],])
    thread_lock.release()
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
    for train_index, test_index in skf.split(X, X[feature]):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]
        if fit_with_val:
            clf.fit(X_train, Y_train, X_val, Y_val)
        else:
            clf.fit(X_train, Y_train)
        pred_val = clf.predict(X_val)
        pred_submit = clf.predict(S) / n_folds
        thread_lock.acquire()
        train_preds[model_name].iloc[test_index] = pred_val
        submit_preds[model_name] += pred_submit
        thread_lock.release()
    train_preds[model_name] = untransform(train_preds[model_name], 
        transformation)
    submit_preds[model_name] = untransform(submit_preds[model_name],
        transformation)
    score = rmsdl(Y_orig, train_preds[model_name])
    if scores_dict is not None:
        scores_dict[model_name] = score
    if score > min_score:
        thread_lock.acquire()
        train_preds.pop(model_name)
        submit_preds.pop(model_name)
        thread_lock.release()
        #print('%s discarded!!!: RMSD score on training data: %0.4f' % (
        #    model_name, score))
        #print('%s: Terminates training %s' % (str(threading.currentThread().getName()),
        #    model_name))
        return score
    
    #print('%s: RMSD score on training data: %0.4f' % (model_name,
    #        score))
    #print('%s: Terminates training %s' % (str(threading.currentThread().getName()),
    #    model_name))
    return score

def train_model_mean(model_name, clf, X, Y_orig, S, train_preds, submit_preds,
            n_folds, min_score, thread_lock, random_state,
            fit_with_val=False, scores_dict=None, transformation=None):
    Y = transform(Y_orig, transformation)
    thread_lock.acquire()
    train_preds[model_name] = np.zeros([X.shape[0],])
    submit_preds[model_name] = np.zeros([S.shape[0],])
    thread_lock.release()
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
    for train_index, test_index in skf_reg_split(skf, X, Y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]
        if fit_with_val:
            clf.fit(X_train, Y_train, X_val, Y_val)
        else:
            clf.fit(X_train, Y_train)
        pred_val = clf.predict(X) / n_folds
        pred_submit = clf.predict(S) / n_folds
        thread_lock.acquire()
        train_preds[model_name] += pred_val
        submit_preds[model_name] += pred_submit
        thread_lock.release()
    train_preds[model_name] = untransform(train_preds[model_name], 
        transformation)
    submit_preds[model_name] = untransform(submit_preds[model_name],
        transformation)
    score = rmsdl(Y_orig, train_preds[model_name])
    if scores_dict is not None:
        scores_dict[model_name] = score
    if score > min_score:
        thread_lock.acquire()
        train_preds.pop(model_name)
        submit_preds.pop(model_name)
        thread_lock.release()
        #print('%s discarded!!!: RMSD score on training data: %0.4f' % (
        #    model_name, score))
        #print('%s: Terminates training %s' % (str(threading.currentThread().getName()),
        #    model_name))
        return score
    
    #print('%s: RMSD score on training data: %0.4f' % (model_name,
    #        score))
    #print('%s: Terminates training %s' % (str(threading.currentThread().getName()),
    #    model_name))
    return score
def get_layer_average(layer_preds):
    return layer_preds.sum(axis=1)/layer_preds.shape[1]

def combine_features(ds, features):
    ds_combined = ds.copy()
    for comb in itertools.combinations(features, 2):
        feat = 'comb_' + comb[0] + '_mult_' + comb[1]
        ds_combined[feat] = ds_combined[comb[0]] * ds_combined[comb[1]]
    return ds_combined

def merge_feature_values(ds, feature, values):
    copy = ds[feature].copy()
    for value in values[1:]:
        copy[copy == value] = values[0]
    ds[feature] = copy

def get_feature_value_indices(ds, feature, values):
    return ds[ds[feature].isin(values)].index
