from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import pandas as pd
import numpy as np
from os import path
import time
from sklearn.model_selection import cross_val_score

path_erweiter  = path.curdir + '/../../Daten/erweitert/'
path_ergebnis  = path.curdir + '/../../Daten/ergebnis/Performance.csv'

ReadTrainMeasure = pd.read_csv(path_erweiter +'TrainMeasure.csv', sep=';')
ReadTestMeasure = pd.read_csv(path_erweiter +'TrainMeasure.csv', sep=';')

ReadTrainMeasure = ReadTrainMeasure.drop(['P-Geschlecht'], axis=1)
#ReadTrainMeasure = pd.DataFrame(ReadTrainMeasure.values.astype(np.float64))
#le = LabelEncoder()
#dataset['index'] = le.fit_transform(dataset.index)

class validation:

    def __init__(self, learning_model,kflod):
        self.learning_model=learning_model
        self.kflod = kflod
        self.X_train = ReadTrainMeasure.loc[:, ReadTrainMeasure.columns !='P-Altersklasse']
        self.y_train = ReadTrainMeasure['P-Altersklasse']
        self.X_test = ReadTestMeasure.loc[:, ReadTestMeasure.columns !='P-Altersklasse']
        self.y_test = ReadTestMeasure['P-Altersklasse']



    def kreuzvalidierng_model(self):
        # Anfang des Fitting
        startTime = time.time()
        clf = self.learning_model.fit(self.X_train,self.y_train)
        endTime = time.time();
        dauert = endTime - startTime
        #Ende des Fitting
        # Anfang der Kreuzvalidierung
        scores = cross_val_score(clf,self.X_train,self.y_train,scoring='precision_macro', cv = self.kflod)
        # Ende der Kreuzvalidierung
        #print("Accuracy: %0.2f : testFehler: %0.2f , in %0.5f" % (scores.mean(), 1-scores.mean(), dauert))
        return dauert, scores

    def writePerformanceModell(self,ModellParmeter):
        Perform = pd.DataFrame(ModellParmeter, index=[0])
        Perform.to_csv(path_ergebnis, sep=';', mode= 'a', header= False, index=False)
        print(Perform)