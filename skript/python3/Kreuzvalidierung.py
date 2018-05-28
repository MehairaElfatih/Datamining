from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import cross_val_score

path_erweiter  = path.curdir + '/../../Daten/erweitert/'

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
        ResubtiotionFehler=[]
        #anfang des Fitting
        #print(self.y_train)
        le = LabelEncoder()
        self.X_train = le.fit_transform(self.X_train )
        lb = LabelBinarizer()
        self.y_train = lb.fit_transform(self.y_train)
        clf = self.learning_model.fit(self.X_train,self.y_train)
        #Ende des Fitting
        # Anfang der Kreuzvalidierung
        scores = cross_val_score(clf,self.X_train,self.y_train,scoring='precision_macro', cv = self.kflod)
        # Ende der Kreuzvalidierung
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
