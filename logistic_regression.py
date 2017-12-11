# --*-- encoding: utf-8 --*--
import math
import random

'''

norm=l2:
  Loss = 1/2 * C * ||w||2 + Sum[ y*log(f(x)) + (1-y)*log(1-f(x)) ]
  f(x) = 1 / (1 + exp(w * x))
  
'''

class Sample(object):
    def __init__(self):
        self.features = []
        self.label = 0
        self.predict = 0

class MyLogisticRegression(object):
    def __init__(self, norm=None, C=1, learning_rate=0.1, max_round=10000, epsilon = 0.01):
        self.__reset()
        self._norm = norm
        self._C = C
        self._learning_rate = learning_rate
        self._max_round = max_round
        self._epsilon = epsilon

    def __reset(self):
        self._w = []
        self._feature_size = 0

    def _compute(self, sample_list, w):
        f = 0.0
        z = [0.0] * len(sample_list)
        for i in range(0, len(sample_list)):
            sample = sample_list[i]
            for j in range(0, self._feature_size):
                z[i] += w[j] * sample.features[j]
            mz = -z[i] if sample.label == 1 else z[i]
            if mz > 0:
                f += mz + math.log(1 + math.exp(-mz))
            else:
                f += math.log(1 + math.exp(mz))
        f = f / len(sample_list)

        if self._norm == "l2":
            for i in range(0, len(w)):
                f += self._C * w[i] * w[i] / 2

        return f, z

    def _gradient(self, sample_list, z, w):
        g = [0.0] * (self._feature_size)
        D = [0.0] * len(sample_list)
        for i in range(0, len(sample_list)):
            sample = sample_list[i]
            D[i] =  1 / (1 + math.exp(-z[i])) - sample.label
            #print sample.features, w, sample.label, z[i], D[i]
        for i in range(0, self._feature_size):
            for j in range(0, len(sample_list)):
                sample = sample_list[j]
                g[i] += D[j] * sample.features[i]
        if self._norm == "l2":
            for i in range(0, self._feature_size-1):
                g[i] += self._C * w[i]
        return g


    def fit(self, sample_list):
        assert isinstance(sample_list, list)
        assert len(sample_list) > 0

        # 给每个样本加上bias项
        self._feature_size = len(sample_list[0].features) + 1
        assert self._feature_size >= 2
        for sample in sample_list:
            sample.features.append(1)
            assert self._feature_size == len(sample.features)
            assert sample.label == 0 or sample.label == 1

        # 初始化权重
        self._w = [0.0] * self._feature_size

        round = 0
        while round < self._max_round:
            f, z = self._compute(sample_list, self._w)
            g = self._gradient(sample_list, z, self._w)

            round += 1
            print "Round %d Loss: %f" % (round, f)
            if f <= self._epsilon:
                break

            k = self._learning_rate
            f_new = 0.0
            w_new = [0.0] * self._feature_size
            while k >= 0.000001:
                for i in range(0, self._feature_size):
                    w_new[i] = self._w[i] - k * g[i]
                f_new, _ = self._compute(sample_list, w_new)
                print "%f: %f ---> %f" % (k, f, f_new)
                if f_new < f:
                    print "learning rate: ", k
                    break
                k = k / 2
            if f_new >= f:
                break
            self._w = w_new
            print "----------------------------"

    def predict(self, sample_list):
        for sample in sample_list:
            z = 0.0
            if len(sample.features) + 1 == len(self._w):
                sample.features.append(1)
            assert len(sample.features) == len(self._w)
            for i in range(0, len(self._w)):
                z += sample.features[i] * self._w[i]
            f = 0.0
            if z > 0:
                f = 1 / (1 + math.exp(-z))
            else:
                f = 1 - 1 / (1 + math.exp(z))
            if f >= 0.5:
                sample.predict = 1
            else:
                sample.predict = 0

    def score(self, sample_list):
        self.predict(sample_list)
        total = [0.0, 0.0, 0.0]
        correct = [0.0, 0.0, 0.0]
        for i in range(0, len(sample_list)):
            sample = sample_list[i]
            total[0] += 1
            if sample.label == 1:
                total[1] += 1
            else:
                total[2] += 1
            if sample.label == sample.predict:
                correct[0] += 1
                if sample.label == 1:
                    correct[1] += 1
                else:
                    correct[2] += 1
        return correct[0] / total[0], correct[1] / total[1], correct[2] / total[2];



def small_test():
    sample_list = []
    s1 = Sample()
    s1.features = [0, 1]
    s1.label = 1
    sample_list.append(s1)
    s2 = Sample()
    s2.features = [1, 0]
    s2.label = 0
    sample_list.append(s2)
    mylr = MyLogisticRegression(norm="l2", C=0.01, learning_rate=0.1, max_round=500, epsilon=1e-4)
    mylr.fit(sample_list)
    print mylr._w

def main():
    import sys
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    X_train = iris_data.data
    y_train = iris_data.target

    (sample_count, feature_size) = X_train.shape
    print sample_count, feature_size

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)

    from sklearn.linear_model import LogisticRegression
    sklearn_lr = LogisticRegression(penalty="l2", tol=1e-4, C=1.0, max_iter=1000)
    sklearn_lr.fit(X_train, y_train)
    print sklearn_lr.score(X_train, y_train)

    print " ---------------------- "

    sample_list = []
    for i in range(0, sample_count):
        sample = Sample()
        for j in range(0, feature_size):
            sample.features.append(X_train[i,j])
        if y_train[i] == 2:
            sample.label = 1.0
        else:
            sample.label = 0.0
        sample_list.append(sample)
    my_lr = MyLogisticRegression(norm="l2", C=0.01, learning_rate=0.1, max_round=500, epsilon=1e-4)
    my_lr.fit(sample_list)
    print my_lr.score(sample_list)

if __name__ == '__main__':
    main()


