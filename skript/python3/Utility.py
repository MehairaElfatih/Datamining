from builtins import print
from os import path
import pandas as pd
import Explorative as exp
import numpy as np
from sklearn.model_selection import train_test_split

#read files

path_orignalFiles = path.curdir + '/../../Daten/original/'
path_final = path.curdir + '/../../Abbildungen/final/'
path_erweiter = path.curdir + '/../../Daten/erweitert/'
path_ergebnis = path.curdir + '/../../Daten/ergebnis/'


class Utility :
    def __init__(self):
        self.ReadMeasure = pd.read_csv(path_orignalFiles + 'measures.csv', sep=';', decimal=',')
        self.ReadToPedict = pd.read_csv(path_orignalFiles + 'to_predict.csv', sep=';', decimal=',')

    # Split data Test und Train
    def create_test_train_data(self):
        print("test")
        print("Ershetzung von Nan value!")
        dfHeader = self.ReadMeasure.columns[self.ReadMeasure.isnull().any()].tolist()
        for x in dfHeader:
            self.ReadMeasure[x].fillna(self.ReadMeasure[x].mean(), inplace=True)

        """
             Die Dateien müssen so angepasst werden, dass die Wert  von den Features  nicht mehr als String betrachtet würden.
             die Alterabstände  und Geschlechte müssen numkerisiert werden

             P_Geschlecht : m= 1 und w = 0

             P_Altersklasse : '20-29 = 0', '30-39 = 1', '40-49 = 2', '50-59 =3','60-69=4' 

        """

        self.ReadMeasure['P-Geschlecht'] = self.ReadMeasure['P-Geschlecht'].str.replace('m', '1')
        self.ReadMeasure['P-Geschlecht'] = self.ReadMeasure['P-Geschlecht'].str.replace('w', '0')

        self.ReadMeasure['P-Altersklasse'] = self.ReadMeasure['P-Altersklasse'].str.replace('20-29', '0')
        self.ReadMeasure['P-Altersklasse'] = self.ReadMeasure['P-Altersklasse'].str.replace('30-39', '1')
        self.ReadMeasure['P-Altersklasse'] = self.ReadMeasure['P-Altersklasse'].str.replace('40-49', '2')
        self.ReadMeasure['P-Altersklasse'] = self.ReadMeasure['P-Altersklasse'].str.replace('50-59', '3')
        self.ReadMeasure['P-Altersklasse'] = self.ReadMeasure['P-Altersklasse'].str.replace('60-69', '4')

        self.ReadMeasure['P-Altersklasse'] = self.ReadMeasure['P-Altersklasse'].astype(int)
        self.ReadMeasure['P-Geschlecht'] = self.ReadMeasure['P-Geschlecht'].astype(int)
        self.ReadMeasure['L-StanceStride'].astype(float)

        print("########################################__ Anfang des Trennens___################################")
        trainMeasure = self.ReadMeasure.loc[self.ReadMeasure['P-KennungAnonym'].isin(
            [3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149,
             151]) == False]
        testMeasure = self.ReadMeasure.loc[self.ReadMeasure['P-KennungAnonym'].isin(
            [3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149,
             151])]

        trainMeasure = trainMeasure.drop(['P-KennungAnonym'], axis=1)
        testMeasure = testMeasure.drop(['P-KennungAnonym'], axis=1)

        self.ReadMeasure.drop(['P-KennungAnonym'], axis=1)

        print("########################################__ Ende des Trennens___################################")
        return self.ReadMeasure, trainMeasure,testMeasure

    def write_erweiterung_datei(self, df, csv_name):
        df.to_csv(path_erweiter + csv_name, sep=';', index=False, float_format='%.10f')



    def To_Prdicted(self):
        dfHeader = self.ReadToPedict.columns[self.ReadToPedict.isnull().any()].tolist()
        for x in dfHeader:
            self.ReadToPedict[x].fillna(self.ReadToPedict[x].mean(), inplace=True)
       # P_Geschlecht: m = 1   und w = 0

        self.ReadToPedict['P-Geschlecht'] = self.ReadToPedict['P-Geschlecht'].str.replace('m', '1')
        self.ReadToPedict['P-Geschlecht'] = self.ReadToPedict['P-Geschlecht'].str.replace('w', '0')
        #self.ReadToPedict['P-Geschlecht'] = self.ReadMeasure['P-Geschlecht'].astype(int)
        #self.ReadToPedict['L-StanceStride'].astype(float)
        #self.write_erweiterung_datei(self.ReadToPedict, 'to_predict.csv')
        return self.ReadToPedict

    def write_predicted(self,predicted,filesname):


        #print(self.ReadToPedict['P-Geschlecht'].replace('1','m'))


        dfpredicted = pd.DataFrame(predicted,columns=['Predicted_Altersklasse'])
        # P_Altersklasse : '20-29 = 0', '30-39 = 1', '40-49 = 2', '50-59 =3','60-69=4'
        dfpredicted['Predicted_Altersklasse'] = dfpredicted['Predicted_Altersklasse'].replace(0,'20-29')
        dfpredicted['Predicted_Altersklasse'] = dfpredicted['Predicted_Altersklasse'].replace(1,'30-39')
        dfpredicted['Predicted_Altersklasse'] = dfpredicted['Predicted_Altersklasse'].replace(2,'40-49')
        dfpredicted['Predicted_Altersklasse'] = dfpredicted['Predicted_Altersklasse'].replace(3,'50-59')
        dfpredicted['Predicted_Altersklasse'] = dfpredicted['Predicted_Altersklasse'].replace(4,'60-69')

        self.ReadToPedict['Predicted_Altersklasse'] = dfpredicted['Predicted_Altersklasse']

        self.ReadToPedict['P-Geschlecht'] = self.ReadToPedict['P-Geschlecht'].replace('1','m')
        self.ReadToPedict['P-Geschlecht'] = self.ReadToPedict['P-Geschlecht'].replace('0', 'w')

        self.ReadToPedict.to_csv(path_final + filesname, sep=';',float_format='%.9f')


    def writePerformanceModell(self, ModellParmeter, filename):
        Perform = pd.DataFrame(ModellParmeter, index=[0])
        Perform.to_csv(path_ergebnis + filename, sep=';', mode='a', header=False, index=False)
        print(Perform)