import sys
import itertools
import numpy as np
import pandas as pd
from math import log, sqrt
from scipy.stats import skew
from datetime import datetime

if sys.version_info[0] == 3:
    from sklearn.model_selection import StratifiedKFold, KFold
else:
    from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
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
        if type(pred) == list:
            pred = pred[0]
        if pred > 0.0:
            delta = pred.copy()
        delta -= ys[idx]
        sum_ += np.power(delta, 2)
    return sqrt(sum_ / predictions.shape[0])


def rmsdl(ys, predictions):
    if predictions[predictions <= 0].shape[0] != 0:
        return float('inf')
    return rmsd(np.log(ys), np.log(predictions))


def rmsd_sqrt(ys, predictions):
    return rmsdl(ys ** 2, predictions ** 2)


def rmsd_inv(ys, predictions):
    return rmsdl(1000.0 / ys, 1000.0 / predictions)


def rmsd_div1000(ys, predictions):
    return rmsdl(1000.0 * ys, 1000.0 * predictions)


def write_results(ids, preds, score, name, output_dir):
    result = pd.DataFrame(preds, columns=['SalePrice'])
    result["Id"] = ids
    result = result.set_index("Id")

    now = datetime.now()
    sub_file = output_dir + '/submission_%s_%0.4f_%s.csv' % (name,
        score, str(now.strftime("%Y-%m-%d-%H-%M")))
    print(("Writing submission: %s" % sub_file))
    result.to_csv(sub_file, index=True, index_label='Id')


def get_mean_delta(ds):
    return np.mean(ds, axis=0), (np.max(ds, axis=0) - np.min(ds, axis=0))


def feature_normalize(ds, mean, delta):
    return (ds - mean) / delta


# Re-order the categorical features in the training data
def get_feature_order(ds, feature, order_feature):
    medians = ds.groupby([feature])[order_feature].median()
    return sorted(list(medians.keys()), key=lambda x: medians[x])


def encode_cats_order(ds_old, ds_with_price, order_feature):
    ds = ds_old.copy()
    for feature in ds.columns.values:
            ftype = ds[feature].dtype
            if ftype == np.float64 or ftype == np.float32 or \
                    ftype == np.int64 or ftype == np.int32:
                    continue
            order = get_feature_order(ds_with_price, feature, order_feature)
            class_mapping = {v: order.index(v) for v in order}
            ds[feature] = ds[feature].map(class_mapping)
    assert(ds.isnull().sum().sum() == 0)
    return ds


def encode_cats_onehot(ds_old):
    ds = ds_old.copy()
    return pd.get_dummies(ds)


def encode_cats_integers(ds_old):
    ds = ds_old.copy()
    cats = ds.dtypes[ds.dtypes == "object"].index.tolist()
    for feat in cats:
        le = LabelEncoder()
        le.fit(ds[feat].tolist())
        ds[feat] = le.transform(ds[feat].tolist())
    return ds


def add_SoldMonths(ds_old):
    # Transform YrSold and MoSold by the number of months till 2016/01
    age_months = (2016 - ds_old.YrSold) * 12 - ds_old.MoSold + 1
    ds = ds_old.drop(['MoSold', 'YrSold'], axis=1).copy()
    ds['SoldMonths'] = age_months
    return ds


def log_skewed_features(ds, threshold=0.75, verbose=False):
    skewness = ds.apply(lambda x: skew(x))
    skewed_feats = skewness[skewness > threshold]
    if verbose:
        print(("Skewed features above %f: %s" % (threshold, skewed_feats)))
    ds_skew = ds.copy()
    ds_skew[skewed_feats.index] = np.log1p(ds_skew[skewed_feats.index])
    assert(ds_skew.isnull().sum().sum() == 0)
    return ds_skew


def scale_ds(ds):
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(ds), columns=ds.columns,
        index=ds.index)


def skf_reg_split(X, Y, n_folds):
    labels = np.arange(0, n_folds, 1)
    Y_binned = pd.cut(Y, n_folds, labels=labels)
    if sys.version_info[0] == 3:
        skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
        return skf.split(X, Y_binned)
    else:
        skf = StratifiedKFold(Y_binned, n_folds=n_folds, random_state=random_state)
        return list(skf)


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
            fit_with_val=False, scores_dict=None, transformation=None,
            split_feature=None, verbose=False, use_kfold=False):
    Y = transform(Y_orig, transformation)
    thread_lock.acquire()
    train_preds[model_name] = np.zeros([X.shape[0], ])
    submit_preds[model_name] = np.zeros([S.shape[0], ])
    thread_lock.release()
    if use_kfold:
        if sys.version_info[0] == 3:
            splits = KFold(n_splits=n_folds, shuffle=True,
                random_state=random_state).split(X)
        else:
            splits = KFold(X.shape[0], n_folds=n_folds, shuffle=True,
                random_state=random_state)
    elif split_feature is None:
        splits = skf_reg_split(X, Y, n_folds)
    else:
        if sys.version_info[0] == 3:
            skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
            splits = skf.split(X, X[feature])
        else:
            skf = StratifiedKFold(X[feature],n_folds=n_folds, random_state=random_state)
            splits = list(skf)
    for train_index, test_index in splits:
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
        if verbose:
            print(('%s discarded!!!: RMSD score on training data: %0.4f' % (
                model_name, score)))
            print(('%s: Terminates training %s' % (
                str(threading.currentThread().getName()), model_name)))
        return score
    if verbose:
        print(('%s: RMSD score on training data: %0.4f' % (model_name,
                score)))
        print(('%s: Terminates training %s' % (
            str(threading.currentThread().getName()), model_name)))
    return score


def get_layer_mean_preds(layer_preds):
    return layer_preds.mean(axis=1)


def get_layer_median_preds(layer_preds):
    return layer_preds.median(axis=1)


def combine_features(ds, features, operator='prod'):
    ds_combined = ds.copy()
    for comb in itertools.combinations(features, 2):
        feat = 'comb_' + comb[0] + '_' + operator + '_' + comb[1]
        if operator == 'prod':
            ds_combined[feat] = ds_combined[comb[0]] * ds_combined[comb[1]]
        elif operator == 'sum':
            ds_combined[feat] = ds_combined[comb[0]] + ds_combined[comb[1]]
        else:
            raise ValueError('operator should be prod or sum')
    return ds_combined


def merge_feature_values(ds, feature, values):
    copy = ds[feature].copy()
    for value in values[1:]:
        copy[copy == value] = values[0]
    ds[feature] = copy


def get_feature_value_indices(ds, feature, values):
    return ds[ds[feature].isin(values)].index
