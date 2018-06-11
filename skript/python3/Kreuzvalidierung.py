import pandas as pd
from os import path
import time
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

path_erweiter = path.curdir + '/../../Daten/erweitert/'
path_ergebnis = path.curdir + '/../../Daten/ergebnis/Performance.csv'

ReadTrainMeasure = pd.read_csv(path_erweiter + 'TrainMeasure.csv', sep=';')
ReadTestMeasure = pd.read_csv(path_erweiter + 'TestMeasure.csv', sep=';')


# ReadTestMeasure.iloc[:0, 0:94].astype(float,errors='raise')

# ReadTrainMeasure = pd.DataFrame(ReadTrainMeasure.values.astype(np.float64))
# le = LabelEncoder()
# dataset['index'] = le.fit_transform(dataset.index)

class validation:

    def __init__(self, learning_model, kflod):
        self.learning_model = learning_model
        self.kflod = kflod
        self.X_train = ReadTrainMeasure.loc[:, ReadTrainMeasure.columns != 'P-Altersklasse'].values
        self.y_train = ReadTrainMeasure['P-Altersklasse'].values

    def kreuzvalidierng_model(self):
        # Anfang des Fitting
        # print(self.X_train.dtypes)
        startTime = time.time()

        clf = self.learning_model.fit(self.X_train, self.y_train)
        endTime = time.time()
        dauert = endTime - startTime
        # Ende des Fitting
        # Anfang der Kreuzvalidierung
        scores = cross_val_score(clf, self.X_train, self.y_train, scoring='accuracy', cv=self.kflod)
        # Ende der Kreuzvalidierung
        # print("Accuracy: %0.2f : testFehler: %0.2f , in %0.5f" % (scores.mean(), 1-scores.mean(), dauert))
        return dauert, scores

    def generated_model(self):

        ReadTrainMeasure = pd.read_csv(path_erweiter + 'TrainMeasure.csv', sep=';', decimal=',')
        ReadTrainMeasure = ReadTrainMeasure.drop(['P-KennungAnonym'], axis=1)
        train, test = train_test_split(ReadTrainMeasure, test_size=0.2)
        X_train = train.loc[:, train.columns != 'P-Altersklasse']
        y_train = train['P-Altersklasse']
        clf = self.learning_model.fit(X_train, y_train)
        scores = clf.predict(test)


    def writePerformanceModell(self, ModellParmeter):
        Perform = pd.DataFrame(ModellParmeter, index=[0])
        Perform.to_csv(path_ergebnis, sep=';', mode='a', header=False, index=False)

        print(Perform)
