from features.feature_set import FeatureSet


class DateFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        features['WeekOfYear'] = data_set.Date.dt.weekofyear.astype(float)
