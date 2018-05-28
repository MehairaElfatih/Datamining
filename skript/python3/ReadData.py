from builtins import print
from os import path
import pandas as pd
import csv

#read files

path_orignalFiles = path.curdir + '/../../Daten/original/'

path_erweiter  = path.curdir + '/../../Daten/erweitert/'

ReadMeasure = pd.read_csv(path_erweiter +'measures_Formatiert.csv', sep=';')
ReadToPedict = pd.read_csv(path_orignalFiles +'to_predict.csv', sep=';')
ReadTrainMeasure = pd.read_csv(path_erweiter +'TrainMeasure.csv', sep=';')
ReadTestMeasure = pd.read_csv(path_erweiter +'TestMeasure.csv', sep=';')

# Split data Test und Train
def create_test_train_data():
    ReadMeasure = Datatype_umwandeln_to_floaot_and_int()
    trainMeasure = ReadMeasure.loc[ReadMeasure['P-KennungAnonym'].isin([3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151])]
    testMeasure = ReadMeasure.loc[ReadMeasure['P-KennungAnonym'].isin([3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151]) == False]
    trainMeasure.to_csv(path_erweiter + 'TrainMeasure.csv', sep=';', encoding='utf-8')
    testMeasure.to_csv(path_erweiter + 'TestMeasure.csv', sep=';', encoding='utf-8')

"""
 Die Dateien müssen so angepasst werden, dass die Wert  von den Features  nicht mehr als String betrachtet würden.
 die Alterabstände  und Geschlechte müssen numkerisiert werden
 
 P_Geschlecht : m= 1 und w = 0
 
 P_Altersklasse : '20-29 = 0', '30-39 = 1', '40-49 = 2', '50-59 =3','60-69=4' 
   
"""

def Datatype_umwandeln_to_floaot_and_int():
    print("########################################__ Anfeng der Umwandelung___################################")
    listcolumns_to_float = list(ReadMeasure.iloc[:0, 0:94])
    for elt in listcolumns_to_float:
         ReadMeasure[elt]= ReadMeasure[elt].str.replace(',','.').astype(float)

    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].str.replace('m','1')
    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].str.replace('w','0')

    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('20-29', '0')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('30-39', '1')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('40-49', '2')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('50-59', '3')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('60-69', '4')

    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].astype(int)
    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].astype(int)

    print ("########################################__ Ende der Umawndelung___################################")
    #print(ReadMeasure)
    return ReadMeasure

def Datatype_wiedererstellung(df):
    print("########################################__ Anfeng der Wiederersellung___################################")
    listcolumns_to_float = list(ReadMeasure.iloc[:0, 0:94])

    for elt in listcolumns_to_float:
         ReadMeasure[elt]= ReadMeasure[elt].astype(str)
         ReadMeasure[elt] = ReadMeasure[elt].str.replace('.', ',')

    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].replace(1,'m')
    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].replace(0,'w')

    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].replace(0,'20-29')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].replace(1,'30-39')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].replace(2,'40-49')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].replace(3,'50-59')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].replace(4,'60-69')
    print("###########################################_______ Ende der Wiedererstellung________###################")
    # print(ReadMeasure)
    return  ReadMeasure


create_test_train_data()
