import numpy as np

from features.feature_set import FeatureSet


class CompetitionFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        self._rewrite(data_set, features, [
            "CompetitionDistance",
        ], data_type=float)

        years_diff = (data_set.Date.dt.year - data_set.CompetitionOpenSinceYear)
        month_diff = (data_set.Date.dt.month - data_set.CompetitionOpenSinceMonth)
        features['CompetitionOpen'] = 12 * years_diff + month_diff
        # features['CompetitionOpen'] = features.CompetitionOpen.astype(float)
        features['CompetitionToOpen'] = features.CompetitionOpen.apply(self._competition_days_to_open).astype(float)
        features['CompetitionFromOpen'] = features.CompetitionOpen.apply(self._competition_days_from_open).astype(float)
        del features["CompetitionOpen"]
        # features['CompetitionOpen'] = features.CompetitionOpen.apply(lambda x: x if x > 0 else 0).astype(float)

    def _competition_days_to_open(self, days_diff):
        if not np.isnan(days_diff):
            return -min(days_diff, 0)
        return np.nan

    def _competition_days_from_open(self, days_diff):
        if not np.isnan(days_diff):
            return max(days_diff, 0)
        return np.nan