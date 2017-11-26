import math
import random

class Sample(object):
    def __init__(self, features=[], lables=[], predicts=[]):
        self.features = features
        self.labels = lables
        self.predicts = predicts

    def debug(self):
        print self.features, self.labels, self.predicts

class PlainKernel(object):
    def compute(self, x):
        return x

    def derive(self, x):
        return 1

class SigmoidKernel(object):
    def compute(self, x):
        return 1 / (1 + math.exp(-x))

    def derive(self, x):
        return self.compute(x) * (1 - self.compute(x))


class SoftplusKernel(object):
    def compute(self, x):
        return math.log(1 + math.exp(x))

    def derive(self, x):
        return 1 / (1 + math.exp(-x))


class ReLUKernel(object):
    def compute(self, x):
        return x if x > 0 else 0

    def derive(self, x):
        return 1 if x > 0 else 0

class PReLUKernel(object):
    def __init__(self):
        self.__alpha = 0.01

    def compute(self, x):
        return x if x > 0 else self.__alpha * x

    def derive(self, x):
        return 1 if x > 0 else self.__alpha

class SquareLoss(object):
    def __init__(self):
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def accumulate(self, olist, ylist):
        for i in range(0, len(olist)):
            self.value += math.pow(olist[i] - ylist[i], 2)

    def derive(self, olist, ylist, n):
        return olist[n] - ylist[n]


class Node(object):
    def __init__(self, kernel=None):
        self.input_value = 0.0
        self.output_value = 0.0
        self.input_derivative = 0.0
        self.output_derivative = 0.0
        self.kernel = kernel

    def reset(self):
        self.input_value = 0.0
        self.output_value = 0.0
        self.input_derivative = 0.0
        self.output_derivative = 0.0

    def compute(self):
        self.output_value = self.kernel.compute(self.input_value)

    def derive(self):
        self.input_derivative = self.output_derivative * self.kernel.derive(self.input_value)

    def debug(self):
        print "[%f,%f; %f,%f] " % (self.input_value, self.output_value, self.input_derivative, self.output_derivative),


class Layer(object):
    def __init__(self, size, kernel):
        self.nodes = [Node(kernel) for i in range(0, size)]

    def reset(self):
        for node in self.nodes:
            node.reset()

    def debug(self):
        for node in self.nodes:
            node.debug()
        print "\n"


class Weight(object):
    def __init__(self, row, column, learning_rate):
        self.row = row
        self.column = column
        self.learning_rate = learning_rate
        self.weight = [[0.0 for c in range(0, self.column)] for r in range(0, self.row)]
        self.derivative = [[0.0 for c in range(0, self.column)] for r in range(0, self.row)]

    def random_init(self):
        for r in range(0, self.row):
            for c in range(0, self.column):
                self.weight[r][c] = random.random()

    def reset_derivative(self):
        for r in range(0, self.row):
            for c in range(0, self.column):
                self.derivative[r][c] = 0.0

    def update_weight(self):
        for r in range(0, self.row):
            for c in range(0, self.column):
                self.weight[r][c] -= self.derivative[r][c] * self.learning_rate
                #self.derivative[r][c] = 0.0

    def debug(self):
        for r in range(0, self.row):
            for c in range(0, self.column):
                print "[%f;%f] " % (self.weight[r][c], self.derivative[r][c]),
            print
        print


class NeuralNetwork(object):
    def __init__(self, max_train_round=10000, min_loss_value=1e-2,
                 learning_rate=1e-1, kernel_functor=SigmoidKernel(),
                 loss_functor=SquareLoss()):
        self.layers = None
        self.weights = None
        self.max_train_round = max_train_round
        self.min_loss_value = min_loss_value
        self.learning_rate = learning_rate
        self.kernel_functor = kernel_functor
        self.loss_functor = loss_functor

    def prepare(self, params):
        self.layers = []
        self.weights = []
        assert isinstance(params, list)
        assert len(params) >= 2
        for i in range(0, len(params)):
            assert params[i] >= 1
            if i == 0:
                layer = Layer(params[i], PlainKernel())
                self.layers.append(layer)
            else:
                layer = Layer(params[i], self.kernel_functor)
                self.layers.append(layer)
            if i + 1 < len(params):
                weight = Weight(params[i]+1, params[i+1], self.learning_rate)
                weight.random_init()
                self.weights.append(weight)

    def train(self, samples):
        train_round = 0
        while True:
            self.loss_functor.reset()
            for sample in samples:
                self._reset_layers()
                self._reset_weight_derivative()
                self._forward_compute(sample)
                self._backward_compute(sample)
                self._update_weight()
            if train_round % 1000 == 0:
                print "round %d loss %f" % (train_round, self.loss_functor.value)
            train_round += 1
            if train_round >= self.max_train_round:
                break
            if self.loss_functor.value <= self.min_loss_value:
                break
        print "round %d loss %f" % (train_round, self.loss_functor.value)

    def predict(self, samples):
        for sample in samples:
            self._reset_layers();
            self._forward_compute(sample)
            for node in self.layers[-1].nodes:
                sample.predicts.append(node.output_value)

    def _reset_layers(self):
        for layer in self.layers:
            layer.reset()

    def _reset_weight_derivative(self):
        for weight in self.weights:
            weight.reset_derivative()

    def _forward_compute(self, sample):
        input_layer = self.layers[0]
        for i in range(0, len(sample.features)):
            input_layer.nodes[i].input_value = sample.features[i]
            input_layer.nodes[i].compute()
        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            before_layer = self.layers[i-1]
            current_weight = self.weights[i-1]
            for n in range(0, len(current_layer.nodes)):
                current_node = current_layer.nodes[n]
                for m in range(0, len(before_layer.nodes)):
                    before_node = before_layer.nodes[m]
                    current_node.input_value += before_node.output_value * current_weight.weight[m][n]
                current_node.input_value += current_weight.weight[len(before_layer.nodes)][n]
                current_node.compute()
        olist = [node.output_value for node in self.layers[-1].nodes]
        self.loss_functor.accumulate(olist, sample.labels)

    def _backward_compute(self, sample):
        output_layer = self.layers[-1]
        olist = [node.output_value for node in output_layer.nodes]
        for m in range(0, len(output_layer.nodes)):
            node = output_layer.nodes[m]
            node.output_derivative = self.loss_functor.derive(olist, sample.labels, m)
            node.derive()
        for i in range(0, len(self.layers)-1)[::-1]:
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            current_weight = self.weights[i]
            for m in range(0, len(current_layer.nodes)):
                current_node = current_layer.nodes[m]
                for n in range(0, len(next_layer.nodes)):
                    next_node = next_layer.nodes[n]
                    current_weight.derivative[m][n] += current_node.output_value * next_node.input_derivative
                    current_node.output_derivative += current_weight.weight[m][n] * next_node.input_derivative
                current_node.derive()
            for n in range(0, len(next_layer.nodes)):
                next_node = next_layer.nodes[n]
                current_weight.derivative[len(current_layer.nodes)][n] = next_node.input_derivative

    def _update_weight(self):
        for i in range(0, len(self.weights)):
            current_weight = self.weights[i]
            current_weight.update_weight()

    def debug(self):
        for i in range(0, len(self.layers)):
            self.layers[i].debug()
            if i < len(self.weights):
                self.weights[i].debug()


def make_training_data():
    sample_list = []
    sample_list.append(Sample([0, 0], [0], []))
    sample_list.append(Sample([0, 1], [1], []))
    sample_list.append(Sample([1, 0], [1], []))
    sample_list.append(Sample([1, 1], [0], []))
    return sample_list

def main():
    sample_list = make_training_data()

    neural_network = NeuralNetwork(
        max_train_round=500000,
        min_loss_value=1e-3,
        learning_rate=1e-4,
        kernel_functor=PReLUKernel(),
        loss_functor=SquareLoss())
    neural_network.prepare([2,4,1])
    neural_network.train(sample_list)

    neural_network.predict(sample_list)
    for sample in sample_list:
        sample.debug()

if __name__ == '__main__':
    main()