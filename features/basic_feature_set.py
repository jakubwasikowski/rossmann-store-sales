from features.feature_set import FeatureSet


class BasicFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        self._rewrite(data_set, features, [
            "SchoolHoliday",
        ], data_type=float)
        self._rewrite(data_set, features, [
            "StoreType",
            "Assortment",
            "StateHoliday",
            "Store",
        ], data_type="category")
