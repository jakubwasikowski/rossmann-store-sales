from features.feature_set import FeatureSet


class BasicFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        self._rewrite(data_set, features, [
            "Store",
            "DayOfWeek",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
            "StoreType_numeric",
            "Assortment_numeric",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval_numeric",
            "Month",
            "Year",
            "Day",
        ], data_type=float)
