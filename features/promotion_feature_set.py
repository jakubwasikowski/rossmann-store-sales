from features.feature_set import FeatureSet


class PromotionFeatureSet(FeatureSet):
    def generate_features(self, data_set, features):
        self._rewrite(data_set, features, [
            "Promo",
            "Promo2",
        ], data_type=float)
        # months_indexes = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sept": 9,
        #                   "Oct": 10, "Nov": 11, "Dec": 12}
        #
        years_diff = (data_set.Date.dt.year - data_set.Promo2SinceYear)
        week_of_year_diff = (data_set.Date.dt.weekofyear - data_set.Promo2SinceWeek)
        features["Promo2Started"] = 53 * years_diff + week_of_year_diff
        features["Promo2Started"] = features.Promo2Started.apply(lambda x: x > 0).astype(float)
        #
        # features["IsPromoMonth"] = 0
        # for interval in data_set.PromoInterval.unique():
        #     if interval != 0:
        #         for month_name in interval.split(','):
        #             promo_month = data_set.Date.dt.month == months_indexes[month_name]
        #             correct_interval = data_set.PromoInterval == interval
        #             promo_started = features.PromoStarted
        #             features.loc[promo_month & correct_interval & promo_started, "IsPromoMonth"] = 1
        # features.drop("PromoStarted", 1)
