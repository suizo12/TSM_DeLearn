import numpy as np


def prepare_data(n, n1, a, s, width=0.6, eps=0.5):
    """
    Generates a random linearly separable 2D test set and associated labels (0|1).
    The x-values are distributed in the interval [-0.5,0.5].
    With the parameters a,s you can control the line that separates the two classes.
    This turns out to be the line with the widest corridor.
    If the random seed is set, the set will always look the same for given input parameters.

    Arguments:
    a -- y-intercept of the seperating line
    s -- slope of the separating line
    n -- size of the sample
    n1 -- number of samples labelled with '1'
    width -- width of the corridor between the two classes
    eps -- measure for the variation of the samples in x2-direction

    Returns:
    x -- generated 2D data of shape (2,n)
    y -- labels of shape (n,)
    """
    np.random.seed(1)
    idx = np.random.choice(n, n1, replace=False)
    y = np.zeros(n, dtype=int)
    y[idx] = 1

    x = np.random.rand(2, n).reshape(2, n)
    x[0, :] -= 0.5
    idx1 = y == 1
    idx2 = y == 0
    x[1, idx1] = (a + s * x[0, idx1]) + (width / 2 + eps * x[1, idx1])
    x[1, idx2] = (a + s * x[0, idx2]) - (width / 2 + eps * x[1, idx2])

    return x, y


import matplotlib.pyplot as plt

n0 = 100


def line(a, s):
    """
    Returns a line 2D array with x and y=a+s*x
    """
    x0 = np.linspace(-0.5, 0.5, n0)
    l = np.array([x0, a + s * x0]).T.reshape(n0, 2)
    return l


def plot(x, y, params_best=None, params_before=None, params_after=None, misclassified=None, selected=None):
    """
    Plot the 2D data provided in form of the x-array.
    Use markers depending on the label ('1 - red cross, 0 - blue cross').
    Optionally, you can pass tuples with parameters for a line (a: y-intercept, s: slope)
    * params_best: ideal separating line (green dashed)
    * params: predicted line (magenta)
    Finally, you can also mark single points:
    * misclassified: array of misclassified points (blue circles)
    * selected: array of selected points (green filled circles)

    Parameters:
    x -- 2D input dataset of shape (2,n)
    y -- ground truth labels of shape (n,)
    params_best -- parameters for the best separating line
    params -- any line parameters
    misclassified -- array of points to be marked as misclassified
    selected -- array of points to be marked as selected
    """
    idx1 = y == 1
    idx2 = y == 0
    plt.plot(x[0, idx1], x[1, idx1], 'r+', label="label 1")
    plt.plot(x[0, idx2], x[1, idx2], 'b+', label="label 0")
    if not params_best is None:
        a = params_best[0]
        s = params_best[1]
        l = line(a, s)
        plt.plot(l[:, 0], l[:, 1], 'g--')
    if not params_before is None:
        a = params_before[0]
        s = params_before[1]
        l = line(a, s)
        plt.plot(l[:, 0], l[:, 1], 'm--')
    if not params_after is None:
        a = params_after[0]
        s = params_after[1]
        l = line(a, s)
        plt.plot(l[:, 0], l[:, 1], 'm-')
    if not misclassified is None:
        plt.plot(x[0, misclassified], x[1, misclassified], 'o', label="misclassified")
    if not selected is None:
        plt.plot(x[0, selected], x[1, selected], 'oy', label="selected")

    plt.legend()
    plt.show()

def lineparams(weight, bias):
    """
    Translates the weights vector and the bias into line parameters with a x2-intercept 'a' and a slope 's'.

    Parameters:
    w -- weights vector of shape (1,2)
    b -- bias (a number)

    Returns:
    a -- x2-intercept
    s -- slope of the line in the (x1,x2)-plane
    """
    ### START YOUR CODE ###
    w1 = weight[0][0]
    w2 = weight[0][1]
    a = -(bias/w2)
    s = -(w1/w2)
    #s = 0 if weight[0][0] == 0 else weight[0][1] / weight[0][0]
    #a = bias
    ### END YOUR CODE ###
    return a, s


def predict(x, w, b):
    """
    Computes the predicted value for a perceptron (single LTU).

    Parameters:
    x -- input dataset of shape (2,...)
    w -- weights vector of shape (1,2)
    b -- bias (a number)

    Returns:
    y -- prediction of a perceptron (single LTU) of shape (1,...)
    """
    ### START YOUR CODE ###
    y = np.matmul(w, x) + b
    y = np.dot(w, x) + b
    #if y2 != y:
    #    raise Exception('not the same y', y, y2)

    y[y >= 0] = 1.
    y[y < 0] = 0.
    return y


def update(x, y, w, b, alpha):
    """
    Performs an update step in accordance with the perceptron learning algorithm.

    Parameters:
    x -- input data point of shape (2,1)
    y -- true label ('ground truth') for the specified point
    w -- weight vector of shape (1,2)
    b -- bias (a number)
    alpha -- learning rate

    Returns:
    w1 -- updated weight vector
    b1 -- updated bias
    """
    ypred = predict(x, w, b)

    ### START YOUR CODE ###
    #w1 = w0 - alpha * x
    #b1 = b - alpha
    if ypred != y:
        w1 = w - x * alpha * (ypred - y)
        b1 = b - alpha * (ypred - y)

        return w1, b1
    ### END YOUR CODE ###

    return w, b


def select_datapoint(x, y, w, b):
    """
    Identifies the misclassified data points and selects one of them (the first in the list).
    In case all datapoints are correctly classified None is returned.

    Parameters:
    x -- input dataset of shape (2,...)
    y -- ground truth labels of shape (n,)
    w -- weights vector of shape (1,2)
    b -- bias (a number)

    Returns:
    x1 -- one of the wrongly classified datapoint
    y1 -- the associated true label
    misclasssified -- array with indices of wrongly classified datapoints or empty array
    """
    ypred = predict(x, w, b)
    mask_misclassified = (ypred != y)[0]
    misclassified = np.where(mask_misclassified)[0]
    if len(misclassified) > 0:
        x1 = x[:, misclassified[0]]
        y1 = y[misclassified[0]]
        return x1, y1, misclassified
    return None, None, []


max_epochs = 100


def train(weight_init, bias_init, x, y, alpha=1.0, debug=False, params_best=None):
    """
    Trains the perceptron (single LTU) for the given data x and ground truth labels y
    by using the perceptron learning algorithm with learning rate alpha.
    The max number of iterations is limited to 1000.

    Optionally, debug output can be provided in form of plots with showing the effect
    of the update (decision boundary before and after the update) provided at each iteration.

    Parameters:
    weight_init -- weights vector of shape (1,2)
    bias_init -- bias (a number)
    x -- input dataset of shape (2,...)
    y -- ground truth labels of shape (n,)
    alpha -- learning rate
    debug -- flag for whether debug information should be provided for each iteration
    params_best -- needed if debug=True for plotting the true decision boundary

    Returns:
    weight -- trained weights
    bias -- trained bias
    misclassified_counts -- array with the number of misclassifications at each iteration
    """
    weight = weight_init
    bias = bias_init
    misclassified_counts = []
    max_iterations = 1000
    iterations = 0
    while iterations <= max_iterations:
        xIdx, yIdx, misclassified = select_datapoint(x, y, weight, bias)
        if len(misclassified) != 0:
            misclassified_counts.append(len(misclassified))
            weight_old, bias_old = weight, bias
            weight, bias = update(xIdx, yIdx, weight, bias, alpha)
            iterations += 1
            if iterations % 10 == 0:
                print("iter: " + str(iterations) + " missclassifieds: " + str(len(misclassified)))

        else:
            break
        if debug:
            params_after = lineparams(weight, bias)
            params_before = lineparams(weight_old, bias_old)
            plot(x, y, params_best=params_best, params_before=params_before, params_after=params_after,
                 misclassified=misclassified, selected=np.array([misclassified[0]]))


    return weight, bias, misclassified_counts



def weights_and_bias(a,s):
    """
    Computes weights vector and bias from line parameters x2-intercept and slope.
    """
    w1 = - s
    w2 = 1.0
    weight = np.array([w1,w2]).reshape(1,2)
    bias = - a
    return weight, bias


n = 100
n1 = 50
a = 2
s = 1
x,y = prepare_data(n,n1,a,s)

params_best = (a,s)
weight_best, bias_best = weights_and_bias(a, s)
print("weight: ", weight_best, "  bias: ", bias_best)
plot(x,y,params_best=params_best)
a1 = 0
s1 = 0
alpha = 1.0

weight1, bias1 = weights_and_bias(a1,s1)
print("Initial Params: ",weight1,bias1)
params = lineparams(weight1, bias1)
plot(x,y,params_best, params)

weight1,bias1,misclassified_counts = train(weight1, bias1, x, y, debug=False, params_best=params_best)

#weight1,bias1,misclassified_counts = train(weight1, bias1, x, y, debug=True, params_best=params_best)
params = lineparams(weight1, bias1)
print("Iterations: ", len(misclassified_counts)-1)
print("Trained Params: ", weight1,bias1)
plot(x,y, params_best=params_best, params_after=params)

nit = len(misclassified_counts)
it = np.linspace(0,nit,nit)

plt.plot(it, misclassified_counts)
#plt.show()

plt.plot(misclassified_counts)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.show()