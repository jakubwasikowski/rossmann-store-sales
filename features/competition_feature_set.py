from features.feature_set import FeatureSet


class CompetitionFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        years_diff = (data_set.Date.dt.year - data_set.CompetitionOpenSinceYear)
        month_diff = (data_set.Date.dt.month - data_set.CompetitionOpenSinceMonth)
        features['CompetitionOpen'] = 12 * years_diff + month_diff
        features['CompetitionOpen'] = features.CompetitionOpen.apply(lambda x: x if x > 0 else 0).astype(float)
