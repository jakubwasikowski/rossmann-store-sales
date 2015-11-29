import argparse
import numpy as np
from os import path, makedirs

import pandas as pd
import xgboost as xgb
from sklearn import cross_validation
from sklearn.externals import joblib

from features.basic_feature_set import BasicFeatureSet
from features.competition_feature_set import CompetitionFeatureSet
from features.date_feature_set import DateFeatureSet
from features.days_number import DaysNumber
from features.features_extractor import FeaturesExtractor
from features.promotion_feature_set import PromotionFeatureSet
from helpers.cross_validation import non_random_train_test_split

test_filename = 'test.csv'
train_filename = 'train.csv'
store_filename = 'store.csv'
state_holidays_filename = 'state_holidays.csv'
store_states_filename = 'store_states.csv'

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
round_num = 10000

eval_and_test_set_size = 0.2
submission_eval_set_size = 0.012


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


def learning(create_submission, features_extractor, training_set, preds_per_store_path=None):
    print ">> LEARNING"

    print "Splitting data set..."
    if create_submission:
        train, valid = non_random_train_test_split(training_set, test_size=submission_eval_set_size)
    else:
        train, valid = non_random_train_test_split(training_set, test_size=eval_and_test_set_size)
        valid, test = cross_validation.train_test_split(valid, test_size=0.5, random_state=10)

    print "Extracting features for training set..."
    train_x, train_names = features_extractor.extract(train)
    train_y = train.Sales
    d_train = xgb.DMatrix(train_x, label=np.log1p(train_y))

    print "Extracting features for validation set..."
    valid_x, _ = features_extractor.extract(valid, feature_names=train_names)
    valid_y = valid.Sales
    d_valid = xgb.DMatrix(valid_x, label=np.log1p(valid_y))

    print "Feature names: %s" % ', '.join(train_names)
    print "Training xgboost..."

    watch_list = [(d_train, 'train'), (d_valid, 'eval')]
    model = xgb.train(params, d_train, round_num, evals=watch_list, early_stopping_rounds=200, feval=rmspe_xg)

    if create_submission is False:
        print("Validating...")
        test_x, _ = features_extractor.extract(test, feature_names=train_names)
        test_y = test.Sales

        train_probs = model.predict(xgb.DMatrix(test_x))
        indices = train_probs < 0
        train_probs[indices] = 0
        error = rmspe(np.expm1(train_probs), test_y.values)
        print "RMSPE error: %f" % error

    if preds_per_store_path is not None:
        save_predictions_per_store(output_dir=preds_per_store_path,
                                   train_set=train,
                                   train_features=train_x,
                                   valid_set=valid,
                                   valid_features=valid_x,
                                   model=model)

    return model, train_names


def save_predictions_per_store(output_dir, train_set, train_features, valid_set, valid_features, model):
    print ">> SAVING PREDICTIONS PER STORE"
    train = pd.DataFrame(train_set)
    train["PredSales"] = np.expm1(model.predict(xgb.DMatrix(train_features)))
    valid = pd.DataFrame(valid_set)
    valid["PredSales"] = np.expm1(model.predict(xgb.DMatrix(valid_features)))

    train = train.iloc[::-1]
    valid = valid.iloc[::-1]

    for store in train.Store.unique():
        df = train[train.Store == store].append(valid[valid.Store == store])
        output_path = path.join(output_dir, "store_%s.csv" % store)
        df[["Store", "Open", "Promo", "Date", "Sales", "PredSales"]].to_csv(output_path, index=False)


def prediction(output_dir_path, model, feature_names, features_extractor, test_set):
    print ">> PREDICTION"

    print "Extracting features..."
    test_x, _ = features_extractor.extract(test_set, feature_names=feature_names)

    print "Predicting..."
    test_probs = model.predict(xgb.DMatrix(test_x))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({"Id": test_set["Id"], "Sales": np.expm1(test_probs)})
    submission.to_csv(path.join(output_dir_path, "predictions.csv"), index=False)
    joblib.dump(model, path.join(output_dir_path, "xgb_model.pkl"))


def preprocess(training_set, test_set):
    training_set.fillna(0, inplace=True)
    training_set = training_set.loc[training_set["Sales"] > 0]

    test_set.loc[test_set.Open.isnull(), 'Open'] = 1
    test_set.fillna(0, inplace=True)

    return training_set, test_set


def run(input_dir_path, external_dir_path, output_dir_path, preds_per_store_path):

    create_submission = True if output_dir_path is not None else False

    train_path = path.join(input_dir_path, train_filename)
    test_path = path.join(input_dir_path, test_filename)
    store_path = path.join(input_dir_path, store_filename)
    store_states_path = path.join(external_dir_path, store_states_filename)
    state_holidays_path = path.join(external_dir_path, state_holidays_filename)

    training_set = pd.read_csv(train_path, parse_dates=[2])
    test_set = pd.read_csv(test_path, parse_dates=[3])
    store = pd.read_csv(store_path)

    training_set = pd.merge(training_set, store, on="Store", how='left')
    test_set = pd.merge(test_set, store, on="Store", how='left')

    features_extractor = FeaturesExtractor()
    features_extractor.add_feature_set(BasicFeatureSet())
    features_extractor.add_feature_set(DateFeatureSet())
    features_extractor.add_feature_set(CompetitionFeatureSet())
    features_extractor.add_feature_set(PromotionFeatureSet())
    features_extractor.add_feature_set(DaysNumber(store_states_path, state_holidays_path))

    training_set, test_set = preprocess(training_set, test_set)
    model, feature_names = learning(create_submission, features_extractor, training_set, preds_per_store_path)
    if create_submission:
        prediction(output_dir_path, model, feature_names, features_extractor, test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data")
    parser.add_argument("--external_dir", default="external")
    parser.add_argument("--output_dir")
    parser.add_argument("--preds_per_store_dir")

    args = parser.parse_args()

    if args.output_dir is not None and not path.exists(args.output_dir):
        makedirs(args.output_dir)

    run(input_dir_path=args.input_dir,
        external_dir_path=args.external_dir,
        output_dir_path=args.output_dir,
        preds_per_store_path=args.preds_per_store_dir)
