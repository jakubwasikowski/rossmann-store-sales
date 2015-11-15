import argparse
from os import path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation

from cross_validation import non_random_train_test_split
from features import FeaturesExtractor, BasicFeatureSet, DateFeatureSet


# params = {
#     "objective": "reg:linear",
#     "eta": 0.2,
#     "min_child_weight": 4,
#     "gamma": 5.0,
#     "max_depth": 8,
#     "subsample": 1.0,
#     "colsample_bytree": 1.0,
#     "silent": 1
# }

params = {
    "objective": "reg:linear",
    "booster": "gbtree",
    "eta": 0.1,
    "max_depth": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.4,
    "min_child_weight": 6,
    "silent": 1,
    "seed": 1301
}
round_num = 1400


def rmspe(yhat, y):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(yhat, y)


# def median(training_set):
#     columns = ['Store', 'DayOfWeek', 'Promo']
#
#     x_train, x_test, y_train, y_test = non_random_train_test_split(training_set, training_set.Sales, test_size=0.0388)
#
#     medians = x_train.groupby(columns)['Sales'].median()
#     medians = medians.reset_index()
#
#     test2 = pd.merge(x_test, medians, on=columns, how='left')
#     assert(len(test2) == len(x_test))
#
#     test2.loc[test2.Open == 0, 'Sales'] = 0
#
#     error = calc_rmspe(test2.Sales.values, y_test.values)
#     print error


def learning(features_extractor, training_set):
    print ">> LEARNING"

    training_set.fillna(0, inplace=True)
    training_set = training_set.loc[training_set["Sales"] > 0]

    print "Splitting data set..."
    # train, valid = non_random_train_test_split(training_set, test_size=0.0388)
    train, valid = cross_validation.train_test_split(training_set, test_size=0.012, random_state=10)

    print "Extracting features..."
    train_x, train_names = features_extractor.extract(train)
    train_y = train.Sales

    valid_x, _ = features_extractor.extract(valid, feature_names=train_names)
    valid_y = valid.Sales

    print "Feature names: %s" % ', '.join(train_names)

    print "Training xgboost..."
    d_train = xgb.DMatrix(train_x, label=np.log1p(train_y))
    d_valid = xgb.DMatrix(valid_x, label=np.log1p(valid_y))

    watch_list = [(d_valid, 'eval'), (d_train, 'train')]
    model = xgb.train(params, d_train, round_num, evals=watch_list, early_stopping_rounds=200, feval=rmspe_xg)

    print("Validating...")
    train_probs = model.predict(xgb.DMatrix(valid_x))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.expm1(train_probs), valid_y.values)
    print "RMSPE error: %f" % error

    return model, train_names


def prediction(out_prediction_path, model, feature_names, features_extractor, test_set):
    print ">> PREDICTION"

    test_set.loc[test_set.Open.isnull(), 'Open'] = 1

    print "Extracting features..."
    test_x, _ = features_extractor.extract(test_set, feature_names=feature_names)

    print "Predicting..."
    test_probs = model.predict(xgb.DMatrix(test_x))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({"Id": test_set["Id"], "Sales": np.expm1(test_probs)})
    submission.to_csv(out_prediction_path, index=False)


def run(out_prediction_path, train_path, test_path, store_path):
    training_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    store = pd.read_csv(store_path)

    training_set = pd.merge(training_set, store, on="Store")
    test_set = pd.merge(test_set, store, on="Store")

    features_extractor = FeaturesExtractor()
    features_extractor.add_feature_set(BasicFeatureSet())
    features_extractor.add_feature_set(DateFeatureSet())

    model, feature_names = learning(features_extractor, training_set)
    prediction(out_prediction_path, model, feature_names, features_extractor, test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("input_dir")

    args = parser.parse_args()

    run(out_prediction_path=path.join(args.output_dir, "predictions.csv"),
        train_path=path.join(args.input_dir, "train.csv"),
        test_path=path.join(args.input_dir, "test.csv"),
        store_path=path.join(args.input_dir, "store.csv"))
