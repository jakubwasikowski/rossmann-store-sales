import operator
import os

import matplotlib.pyplot as plt
import pandas as pd


_fmap_path = "xgb.fmap"
_fimp_path = "feature_importance_xgb.png"


def plot_feature_importance(model, feature_names):
    _create_feature_map(feature_names)
    importance = model.get_fscore(fmap=_fmap_path)
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    os.remove(_fmap_path)

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig(_fimp_path, bbox_inches='tight', pad_inches=1)


def _create_feature_map(features):
    outfile = open(_fmap_path, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()