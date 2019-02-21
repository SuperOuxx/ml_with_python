# 可视化iris数据集

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
# features = np.ndarray(data['data']).reshape(150, 4) # , (150, 4))
features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
# 是setosa
is_setosa = (target == 0)
# 是virginica
virginica = (target == 2)

def visual(features, target):
    for t, marker, c in zip(range(3), ">ox", 'rgb'):
        plt.scatter(features[target == t, 0],
                    features[target == t, 1],
                    marker=marker,
                    c=c)
    plt.show()

def diff_setosa():
    """
    用花瓣宽度来区分setosa与非setosa
    :return:
    """
    plength = features[:, 2]

    # setosa中花瓣长度最大值
    max_setosa = np.max(plength[is_setosa])
    # 非setosa花中的花瓣长度的最小值
    min_non_setosa = np.min(plength[~is_setosa])
    print('Max of setosa: ', max_setosa)
    print('Min of others: ', min_non_setosa)


def diff():
    # 只选择非setosa
    non_setosa_features = features[~is_setosa]
    non_setosa_targets = target[~is_setosa]

    best_acc = -1.0
    for fi in range(np.mat(non_setosa_features).shape[1]):
        thresh = np.copy(non_setosa_features[:, fi])
        thresh.sort()
        # 测试所有阈值
        for t in thresh:
            pred = (non_setosa_features[:, fi] > t)
            acc = np.mean(pred == virginica)
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
    print('best_fi', best_fi, 'best_t', best_t)


def classify_with_sklearn(features):
    """
    用sklearn作分类
    :param features:
    :return:
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
    kf = KFold(len(features), n_folds=5, shuffle=True)
    # `means` will be a list of mean accuracies (one entry per fold)
    means = []
    for train, test in kf:
        classifier.fit(features[train], target[train])
        prediction = classifier.predict(features[test])
        print('test : ', target[test])
        print('train : ', target[train])
        # np.mean on an array of booleans returns fraction
        # of correct decisions for this fold:
        curmean = np.mean(prediction == target[test])
        means.append(curmean)
    print("Mean accuracy: {:.1%}".format(np.mean(means)))



if __name__ == '__main__':
    # classify_with_sklearn(features)
    # visual(features, target)
    # diff_setosa()
    # diff()
    x = np.array([1, 2, 3])
    print('ndim: {} shape: {} \r\n{}'.format(x.ndim, x.shape, x))

    x1 = x.reshape([-1, 1])
    print('ndim: {} , shape: {} \r\n{}'.format(x1.ndim, x1.shape, x1))

    x2 = x.reshape([1, -1])
    print('ndim: {} , shape: {} \r\n{}'.format(x2.ndim, x2.shape, x2))