import csv
import sys
from datetime import datetime
import pylru
from features.feature_set import FeatureSet


class DaysNumberToHoliday(FeatureSet):
    def __init__(self, store_states_path, state_holidays_path):
        FeatureSet.__init__(self)
        self.store_states_path = store_states_path
        self.state_holidays_path = state_holidays_path

    def generate_features(self, data_set, features):
        print "Extracting days numbers..."
        holidays_per_store, holidays_names = self._get_store_holidays()
        new_columns = self._initialize_new_columns(data_set, holidays_names)
        self._calculate_days_intervals_for_holidays(data_set, holidays_per_store, new_columns)
        self._add_new_columns(features, new_columns)

    def _initialize_new_columns(self, data_set, holidays_names):
        new_columns = {}
        for h_name in holidays_names:
            new_columns["%sPastHoliday" % h_name] = [0] * len(data_set)
            new_columns["%sFutureHoliday" % h_name] = [0] * len(data_set)
        return new_columns

    def _calculate_days_intervals_for_holidays(self, data_set, holidays_per_store, new_columns):
        row_no = len(data_set)
        step = 0
        cache = pylru.lrucache(10000)
        for index, store, timestamp in data_set[["Store", "Date"]].itertuples():
            if (step + 1) % 10000 == 0:
                print "Days number to holiday: %.1f%%" % (float(step) * 100 / row_no)
            for h_name, h_dates in holidays_per_store[store].iteritems():
                if h_dates:
                    if (timestamp, h_dates) in cache:
                        min_past_interval, min_future_interval = cache[(timestamp, h_dates)]
                    else:
                        past_intervals = [(timestamp - h_date).days + 1 for h_date in h_dates if timestamp >= h_date]
                        min_past_interval = min(past_intervals) if past_intervals else None
                        future_intervals = [(h_date - timestamp).days + 1 for h_date in h_dates if timestamp <= h_date]
                        min_future_interval = min(future_intervals) if future_intervals else None
                        cache[(timestamp, h_dates)] = (min_past_interval, min_future_interval)
                    if min_past_interval is not None:
                        new_columns["%sPastHoliday" % h_name][step] = min_past_interval
                    if min_future_interval is not None:
                        new_columns["%sFutureHoliday" % h_name][step] = min_future_interval
            step += 1

    def _add_new_columns(self, features, new_columns):
        for column_name, column_values in new_columns.iteritems():
            features[column_name] = column_values

    def _get_store_holidays(self):
        state_holidays, holidays_names = self._get_state_holidays()
        store_holidays = {}
        with open(self.store_states_path) as store_states_file:
            store_states_reader = csv.reader(store_states_file)
            store_states_reader.next()
            for store_id, state_code in store_states_reader:
                store_holidays[int(store_id)] = state_holidays[state_code]
        return store_holidays, holidays_names

    def _get_state_holidays(self):
        state_holidays = {}
        with open(self.state_holidays_path) as state_holidays_file:
            states_holidays_reader = csv.reader(state_holidays_file)
            header = states_holidays_reader.next()
            holidays_names = header[1:]
            for row in states_holidays_reader:
                state = row[0]
                holidays_dates_str = row[1:]
                state_holidays[state] = \
                    {h_name: self._to_dates(holidays_dates_str[i]) for i, h_name in enumerate(holidays_names)}
        return state_holidays, holidays_names

    def _to_dates(self, holiday_dates_str):
        if holiday_dates_str:
            return tuple(datetime.strptime(date_str, "%Y-%m-%d") for date_str in holiday_dates_str.split(','))
        else:
            return tuple()
