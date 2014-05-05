import numpy
from sklearn import svm

L = numpy.loadtxt(open("train.csv","rb"),delimiter=",")

numpy.random.shuffle(L)

X = L[:,:-1]
y = numpy.ravel(L[:,-1:])

split = int(round(X.shape[0] * 0.7))

Xtrain, Xval = numpy.split(X, [split])
ytrain, yval = numpy.split(y, [split])

print(Xtrain.shape)
print(ytrain.shape)


for c in [0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0]:

	classifier = svm.SVC(C=c, kernel="linear")
	classifier.fit(Xtrain, ytrain)

	print("Trained C=%f train=%f test=%f" % (c, classifier.score(Xtrain, ytrain), classifier.score(Xval, yval)))