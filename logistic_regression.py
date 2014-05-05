import numpy
import scipy
import scipy.io

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load stuff
X = scipy.io.loadmat("mlclass-ex4/ex4data1.mat")["X"]
y = scipy.io.loadmat("mlclass-ex4/ex4data1.mat")["y"]

print("--> Loaded")


# Shuffle examples
L = numpy.zeros((X.shape[0], X.shape[1] + y.shape[1]))
L[:,:-(y.shape[1])] = X
L[:,-(y.shape[1]):] = y

numpy.random.shuffle(L)

X = L[:,:-(y.shape[1])]
y = L[:,-(y.shape[1]):]

print("--> Shuffled")


# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = numpy.ravel(y)

print("--> Preprocessed")


# Split train and test
split = int(round(X.shape[0] * 0.7))

X_train = X[:split,:]
y_train = y[:split]

X_test = X[(split+1):,:]
y_test = y[(split+1):]


# Run logistic regression
print("--> Training l1")
for c in [0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0]:
    lr = LogisticRegression(C=(1/c), penalty='l1', tol=0.1)
    lr.fit(X_train, y_train)
    print("    Trained lambda=%f train=%f test=%f" % (c, lr.score(X_train, y_train), lr.score(X_test, y_test)))

print("--> Training l2")
for c in [0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0]:
    lr = LogisticRegression(C=(1/c), penalty='l2', tol=0.01)
    lr.fit(X_train, y_train)
    print("    Trained lambda=%f train=%f test=%f" % (c, lr.score(X_train, y_train), lr.score(X_test, y_test)))
