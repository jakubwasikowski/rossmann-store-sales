import operator
import os

import matplotlib.pyplot as plt
import pandas as pd


_feature_map_path = "xgb.fmap"
_feature_imp_chart_path = "feature_importance_xgb.png"
_feature_imp_file_path = "feature_importance_xgb.txt"


def plot_feature_importance(model, feature_names):
    _create_feature_map(feature_names)
    importance = model.get_fscore(fmap=_feature_map_path)
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    os.remove(_feature_map_path)

    df = pd.DataFrame(importance[-40:], columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig(_feature_imp_chart_path, bbox_inches='tight', pad_inches=1)

    with open(_feature_imp_file_path, "w") as xgb_imp_file:
        for name, importance in reversed(importance):
            xgb_imp_file.write("%s\t%f\n" % (name, importance))


def _create_feature_map(features):
    outfile = open(_feature_map_path, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()