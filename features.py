import pandas as pd

from encoding import encode_onehot


class FeatureSet:
    def __init__(self):
        pass

    def generate_features(self, data_set, features):
        raise NotImplementedError()

    def _rewrite(self, data_set, features, features_names, data_type):
        for f_name in features_names:
            features[f_name] = data_set[f_name].astype(data_type)


class BasicFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        self._rewrite(data_set, features, ["Promo", "CompetitionDistance", "CompetitionOpenSinceMonth",
                      "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear"], data_type=float)
        # self._rewrite(data_set, features, ["Store", "DayOfWeek", "StoreType", "Assortment"], data_type=float)


class DateFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        features["Year"] = data_set.Date.apply(lambda x: x.split('-')[0]).astype(float)
        features["Month"] = data_set.Date.apply(lambda x: x.split('-')[1]).astype(float)
        features["Day"] = data_set.Date.apply(lambda x: x.split('-')[2]).astype(float)


class FeaturesExtractor:
    def __init__(self):
        self._feature_set_generators = []

    def add_feature_set(self, feature_set):
        self._feature_set_generators.append(feature_set)

    def extract(self, data_set):
        feature_set = pd.DataFrame(index=data_set.index)
        for fs_gen in self._feature_set_generators:
            fs_gen.generate_features(data_set, feature_set)
        #cat_cols = [column for column in feature_set.columns.values if feature_set.dtypes[column].name == "category"]
        #feature_set = encode_onehot(feature_set, cat_cols)

        return feature_set
