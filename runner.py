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
from features.days_number_to_holiday import DaysNumberToHoliday
from features.features_extractor import FeaturesExtractor
from features.promotion_days_number import PromotionDaysNumber
from features.promotion_feature_set import PromotionFeatureSet
from helpers.cross_validation import non_random_train_test_split
from helpers.feature_importance import plot_feature_importance

test_filename = 'R_test_transformed.csv'
train_filename = 'R_train_transformed.csv'
state_holidays_filename = 'state_holidays.csv'
store_states_filename = 'store_states.csv'

params = {
    "objective": "reg:linear",
    "booster": "gbtree",
    "eta": 0.02,
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "silent": 1,
    "seed": 1
}
round_num = 3000

eval_and_test_set_size = 0.2
submission_eval_set_size = 0.012


def rmspe(yhat, y):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(yhat, y)


def learning(create_submission, features_extractor, training_set, preds_per_store_path=None):
    print ">> LEARNING"

    print "Splitting data set..."
    if create_submission:
        train, valid = cross_validation.train_test_split(training_set, test_size=submission_eval_set_size,
                                                         random_state=10)
    else:
        train, valid = cross_validation.train_test_split(training_set, test_size=eval_and_test_set_size,
                                                         random_state=10)
        valid, test = cross_validation.train_test_split(valid, test_size=0.5, random_state=10)

    print train[0:5]
    print "Extracting features for training set..."
    train_x, feature_names = features_extractor.extract(train)
    train_y = train.Sales
    d_train = xgb.DMatrix(train_x, label=np.log1p(train_y))
    print train_x[0:5]
    print "Extracting features for validation set..."
    valid_x, _ = features_extractor.extract(valid)
    valid_y = valid.Sales
    d_valid = xgb.DMatrix(valid_x, label=np.log1p(valid_y))

    print "Training xgboost..."

    watch_list = [(d_train, 'train'), (d_valid, 'eval')]
    model = xgb.train(params, d_train, round_num, evals=watch_list, feval=rmspe_xg)

    if create_submission is False:
        print("Validating...")
        test_x, _ = features_extractor.extract(test)
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

    return model, feature_names


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


def prediction(output_dir_path, model, features_extractor, test_set):
    print ">> PREDICTION"

    print "Extracting features..."
    test_x, _ = features_extractor.extract(test_set)

    print "Predicting..."
    test_probs = model.predict(xgb.DMatrix(test_x))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({"Id": test_set["Id"], "Sales": np.expm1(test_probs)})
    submission.to_csv(path.join(output_dir_path, "predictions.csv"), index=False)
    joblib.dump(model, path.join(output_dir_path, "xgb_model.pkl"))


def run(input_dir_path, external_dir_path, output_dir_path, preds_per_store_path):

    create_submission = True if output_dir_path is not None else False

    train_path = path.join(input_dir_path, train_filename)
    test_path = path.join(input_dir_path, test_filename)

    training_set = pd.read_csv(train_path, parse_dates=[2])
    test_set = pd.read_csv(test_path, parse_dates=[3])

    features_extractor = FeaturesExtractor()
    features_extractor.add_feature_set(BasicFeatureSet())
    features_extractor.add_feature_set(DateFeatureSet())
    features_extractor.add_feature_set(CompetitionFeatureSet())
    # Exclude PromotionFeatureSet due to seemingly poor results
    # features_extractor.add_feature_set(PromotionFeatureSet())


    model, feature_names = learning(create_submission, features_extractor, training_set, preds_per_store_path)
    plot_feature_importance(model, feature_names)
    if create_submission:
        prediction(output_dir_path, model, features_extractor, test_set)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

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
