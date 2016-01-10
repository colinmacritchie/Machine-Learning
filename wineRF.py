import urllib2
import numpy
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

# Read wine quality data from UCI website
target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib2.urlopen(target_url)

xList = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";")
        firstLine = False
    else:
        row = line.strip().split(";")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

nrows = len(xList)
ncols = len(xList[0])

x = numpy.array(xList)
y = numpy.array(labels)
wineNames = numpy.array(names)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.30, random_state=531)

#sizes in order to
mseOos = []
nTreeList = range(50, 500, 10)
for iTrees in nTreeList:
    depth = None
    maxFeat = 4
    wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTree_max_depth,
    max_features=maxFeat,
    obb_score=False, random_state=531)

wineRFModel.fit(xTrain, yTrain)

prediction = wineRFModel.predict(xTest)
mseOos.append(mean_squared_error(yTest, prediction))

print ("MSE")
print (msOos[-1])

trees in ensemble
plot.plot(nTreeList, mseOos)
plot.xLabel("Number of trees in ensemble")
plot.ylabel("mean squared error")
plot.show()

featureImportance = wineRFModel.feature_importances_
featureImportance = featureImportance / featureImportance.max()
sorted_idx = numpy.argsort(featureImportance)
barPos = numpy.arange(sorted_idx.shape[0])
+.5
plot.barh(barPos, featureImportance[sorted_idx],
        align='center')
plot.yticks(barPos, wineNames[sorted_idx])
ploy.xlabel('Variable Importance')
plot.show()

