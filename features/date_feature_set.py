from features.feature_set import FeatureSet


class DateFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        features['Year'] = data_set.Date.dt.year.astype(float)
        features['Month'] = data_set.Date.dt.month.astype(float)
        features['Day'] = data_set.Date.dt.day.astype(float)
        features['DayOfWeek'] = data_set.Date.dt.dayofweek.astype(float)
        features['WeekOfYear'] = data_set.Date.dt.weekofyear.astype(float)
