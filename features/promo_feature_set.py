from collections import defaultdict

from features.feature_set import FeatureSet


class PromoFeatureSet(FeatureSet):
    def __init__(self, train, test):
        FeatureSet.__init__(self)
        self.days_to_promo_start, self.days_after_promo_start, self.days_to_promo_end, self.days_after_promo_end = \
            self._initialize_promo_starts_ends(train, test)

    def generate_features(self, data_set, features):
        self._rewrite(data_set, features, [
            "Promo",
        ], data_type=float)

        col_to_promo_start = []
        col_after_promo_start = []
        col_to_promo_end = []
        col_after_promo_end = []

        row_no = len(data_set)
        step = 0
        for index, store, date in data_set[["Store", "Date"]].itertuples():
            if (step + 1) % 10000 == 0:
                print "Days number to promotion days: %.1f%%" % (float(step) * 100 / row_no)
            col_to_promo_start.append(self.days_to_promo_start[store][date])
            col_after_promo_start.append(self.days_after_promo_start[store][date])
            col_to_promo_end.append(self.days_to_promo_end[store][date])
            col_after_promo_end.append(self.days_after_promo_end[store][date])
            step += 1
        features["DaysToPromoStart"] = col_to_promo_start
        features["DaysAfterPromoStart"] = col_after_promo_start
        features["DaysToPromoEnd"] = col_to_promo_end
        features["DaysAfterPromoEnd"] = col_after_promo_end

    def _initialize_promo_starts_ends(self, train, test):
        print "Initializing promotion starts and ends..."
        df = train[["Store", "Date", "Promo"]].append(test[["Store", "Date", "Promo"]]).sort("Date", ascending=True)

        days_to_promo_start = defaultdict(lambda: defaultdict(int))
        days_after_promo_start = defaultdict(lambda: defaultdict(int))
        days_to_promo_end = defaultdict(lambda: defaultdict(int))
        days_after_promo_end = defaultdict(lambda: defaultdict(int))

        self._calc_days_numbers(days_after_promo_start, days_after_promo_end, df)
        self._calc_days_numbers(days_to_promo_end, days_to_promo_start, df.iloc[::-1])

        return days_to_promo_start, days_after_promo_start, days_to_promo_end, days_after_promo_end

    def _calc_days_numbers(self, days_after_promo_start, days_after_promo_end, df):
        last_date_per_store = {}
        for index, store, date, promo in df.itertuples():
            try:
                last_date = last_date_per_store[store]
                dates_diff = abs((date - last_date).days)
                if promo == 1:
                    days_after_promo_start[store][date] = days_after_promo_start[store][last_date] + dates_diff
                    days_after_promo_end[store][date] = 0
                elif promo == 0:
                    days_after_promo_end[store][date] = days_after_promo_end[store][last_date] + 1
                    days_after_promo_start[store][date] = 0
            except KeyError:
                if promo == 1:
                    days_after_promo_start[store][date] = 1
                    days_after_promo_end[store][date] = 0
                elif promo == 0:
                    days_after_promo_end[store][date] = 1
                    days_after_promo_start[store][date] = 0
            last_date_per_store[store] = date
