import pandas as pd
from os import path
import time
import numpy as np
import Utility as utis
import Explorative as exp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

path_orignalFiles = path.curdir + '/../../Daten/original/'
path_final = path.curdir + '/../../Abbildungen/final/'

path_erweiter = path.curdir + '/../../Daten/erweitert/'
#path_ergebnis = path.curdir + '/../../Daten/ergebnis/Performance.csv'

ReadTrainMeasure = pd.read_csv(path_erweiter + 'TrainMeasure.csv', sep=';')
ReadTestMeasure = pd.read_csv(path_erweiter + 'TestMeasure.csv', sep=';')


test = exp.Explorative()
#ReadTrainMeasure= test.data_normalisation(ReadTrainMeasure)
#ReadTestMeasure = test.data_normalisation(ReadTestMeasure)

#print(ReadTrainMeasure)

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
        test = exp.Explorative()
        startTime = time.time()
        clf = self.learning_model.fit(self.X_train, self.y_train)
        endTime = time.time()
        dauert = endTime - startTime
        # Ende des Fitting
        # Anfang der Kreuzvalidierung
        scores = cross_val_score(clf, self.X_train, self.y_train, scoring='accuracy', cv=self.kflod)
        # Ende der Kreuzvalidierung
        # print("Accuracy: %0.2f : testFehler: %0.2f , in %0.5f" % (scores.mean(), 1-scores.mean(), dauert))
        print(np.mean(scores))
        return dauert, scores

    def generated_model_90(self):
        #ReadTrainMeasure = pd.read_csv(path_erweiter + 'TrainMeasure.csv', sep=';', decimal=',')
        X_train = ReadTrainMeasure.drop(['P-Altersklasse'], axis=1)
        y_train = ReadTrainMeasure['P-Altersklasse']
        startTime = time.time()
        clf = self.learning_model.fit(self.X_train, self.y_train)
        endTime = time.time()
        dauert = endTime - startTime
        #apply Model on Test
        ReadTestMeasure = pd.read_csv(path_erweiter + 'TestMeasure.csv', sep=';', decimal=',')
        X_test = ReadTestMeasure.drop(['P-Altersklasse'], axis=1)
        y_test = ReadTestMeasure['P-Altersklasse']
        #clf = self.learning_model.predict(X_test)

        cm = confusion_matrix(y_test, clf.predict(X_test))
        print(" confusion Matrix 90")
        print(cm)
        # accuracy
        #sum(diag(d)) / sum(d)  # overall accuracy
        #1 - sum(diag(d)) / sum(d)  # incorrect classification
        accuracy = np.sum(np.diagonal(cm))/np.sum(cm)
        return cm, accuracy,dauert

    def generate_predictive(self):
        """
         Generated the with all Data Set
        :return:
        """
        ReadMeasure = pd.read_csv(path_erweiter + 'Measure_Umgewandelt.csv', sep=';', decimal=',')
        ReadMeasure = ReadMeasure.drop(['P-KennungAnonym'], axis=1)
        X_train = ReadMeasure.drop(['P-Altersklasse'], axis=1)
        y_train = ReadMeasure['P-Altersklasse']
        clf = self.learning_model.fit(X_train,y_train)
        ## save predictive Model
        joblib.dump(clf, path_final +'predictive_model.pkl')

    def Predicted_data(self):
        ReadToPredict = pd.read_csv(path_orignalFiles + 'to_predict.csv', sep=';', decimal=',').values
        clf = joblib.load(path_final + 'predictive_model.pkl')

        uti = utis.Utility()
        to_predict = uti.To_Prdicted()

        clf = self.learning_model.predict(to_predict)
        print(clf)
        uti.write_predicted(clf,'my_prediction.csv')
