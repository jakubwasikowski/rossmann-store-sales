import argparse
from os import path

import numpy as np
import pandas as pd
import xgboost as xgb

from cross_validation import non_random_train_test_split
from features import FeaturesExtractor, BasicFeatureSet, DateFeatureSet


params = {
    "objective": "reg:linear",
    "eta": 0.2,
    "min_child_weight": 4,
    "gamma": 5.0,
    "max_depth": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "silent": 1
}

# params = {
#     "objective": "reg:linear",
#     "eta": 0.3,
#     "max_depth": 30,
#     "subsample": 0.7,
#     "colsample_bytree": 0.7,
#     "silent": 1
# }


def calc_rmspe(yhat, y):
    rmspe = np.sqrt(np.mean(((y - yhat) / y)**2))
    return rmspe


def calc_rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    rmspe = np.sqrt(np.mean(((y - yhat) / y)**2))
    return "rmspe", rmspe


def learning(features_extractor, training_set):
    training_set.fillna(0, inplace=True)

    labels = training_set["Sales"]
    features = features_extractor.extract(training_set)

    x_train, x_test, y_train, y_test = non_random_train_test_split(features, labels, test_size=0.0388)

    d_train = xgb.DMatrix(x_train, np.log(y_train + 1))
    d_valid = xgb.DMatrix(x_test, np.log(y_test + 1))

    watch_list = [(d_valid, 'eval'), (d_train, 'train')]
    gbm = xgb.train(params, d_train, 2000, evals=watch_list, early_stopping_rounds=100,
                    feval=calc_rmspe_xg)

    print("Validating")
    train_probs = gbm.predict(xgb.DMatrix(x_test))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = calc_rmspe(np.exp(train_probs) - 1, y_test.values)
    print "RMSPE error: %f" % error


def run(out_prediction_path, train_path, test_path, store_path):
    training_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    store = pd.read_csv(store_path)

    training_set = training_set.loc[training_set["Sales"] > 0]
    test_set.loc[test_set.Open.isnull(), 'Open'] = 1

    training_set = pd.merge(training_set, store, on="Store")
    test_set = pd.merge(test_set, store, on="Store")

    features_extractor = FeaturesExtractor()
    features_extractor.add_feature_set(BasicFeatureSet())
    features_extractor.add_feature_set(DateFeatureSet())

    learning(features_extractor, training_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("input_dir")

    args = parser.parse_args()

    train_path = path.join(args.input_dir, "train.csv")
    test_path = path.join(args.input_dir, "test.csv")
    store_path = path.join(args.input_dir, "store.csv")

    out_prediction_path = path.join(args.output_dir, "predictions.csv")

    run(out_prediction_path, train_path, test_path, store_path)
