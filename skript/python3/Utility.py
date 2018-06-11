from builtins import print
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#read files

path_orignalFiles = path.curdir + '/../../Daten/original/'

path_erweiter  = path.curdir + '/../../Daten/erweitert/'
path_ergebnis  = path.curdir + '/../../Daten/ergebnis/'

ReadMeasure = pd.read_csv(path_orignalFiles +'measures.csv', sep=';', decimal=',')
ReadToPedict = pd.read_csv(path_orignalFiles +'to_predict.csv', sep=';', decimal=',')


# Split data Test und Train
def create_test_train_data():
    print("Ershetzung von Nan value!")
    dfHeader = ReadMeasure.columns[ReadMeasure.isnull().any()].tolist()
    for x in dfHeader:
        ReadMeasure[x].fillna(ReadMeasure[x].mean(), inplace=True)

    for x in dfHeader:
        ReadToPedict[x].fillna(ReadToPedict[x].mean(), inplace=True)

    print(ReadMeasure.loc[:, ReadMeasure.isnull().any()])

    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].str.replace('m', '1')
    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].str.replace('w', '0')

    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('20-29', '0')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('30-39', '1')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('40-49', '2')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('50-59', '3')
    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].str.replace('60-69', '4')

    ReadMeasure['P-Altersklasse'] = ReadMeasure['P-Altersklasse'].astype(float)
    ReadMeasure['P-Geschlecht'] = ReadMeasure['P-Geschlecht'].astype(float)
    #ReadMeasure[:0, 0:94].astype(float)
    ReadMeasure['L-StanceStride'].astype(float)
    #ReadMeasure['L-StanceStride'] = ReadMeasure['L-StanceStride'].str.replace(',', '.')
    #ReadMeasure['L-StanceStride'] = ReadMeasure['L-StanceStride'].apply(pd.to_numeric, errors='coerce')


    print("########################################__ Anfang des Trennens___################################")
    trainMeasure = ReadMeasure.loc[ReadMeasure['P-KennungAnonym'].isin([3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151]) == False]
    testMeasure = ReadMeasure.loc[ReadMeasure['P-KennungAnonym'].isin([3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151])]


    trainMeasure=trainMeasure.drop(['P-KennungAnonym'], axis=1)
    testMeasure = testMeasure.drop(['P-KennungAnonym'], axis=1)

    ReadMeasure.drop(['P-KennungAnonym'], axis=1)

    print("########################################__ Ende des Trennens___################################")

    ReadMeasure.to_csv(path_erweiter + 'Measure_Umgewandelt.csv', sep=';', index = False , float_format='%.5f')
    trainMeasure.to_csv(path_erweiter + 'TrainMeasure.csv', sep=';', index =False, float_format='%.5f')
    testMeasure.to_csv(path_erweiter + 'TestMeasure.csv', sep=';', index =False, float_format='%.5f')







"""
 Die Dateien müssen so angepasst werden, dass die Wert  von den Features  nicht mehr als String betrachtet würden.
 die Alterabstände  und Geschlechte müssen numkerisiert werden
 
 P_Geschlecht : m= 1 und w = 0
 
 P_Altersklasse : '20-29 = 0', '30-39 = 1', '40-49 = 2', '50-59 =3','60-69=4' 
   
"""

def Datatype_umwandeln_to_floaot_and_int():

    print("########################################__ Anfang der Umwandelung___################################")
    listcolumns_to_float = list(ReadMeasure.iloc[:0, 0:94])
    #print(ReadMeasure)
    i = 0
    j = 0
    for elt in listcolumns_to_float:
         try:
             ReadMeasure[elt]= ReadMeasure[elt].str.replace(',','.').astype(float, errors='raise')
             #ReadMeasure.round({elt:2})
             print( elt + " is  good!")
             i = i+1
         except ValueError:
             print(elt + " '''''''''''''''''''''''''''''''''''  is  not  good!")
             ReadMeasure[elt] = ReadMeasure[elt].str.replace(',', '.')
             ReadMeasure[elt] = ReadMeasure[elt].astype(float, errors='ignore')
             #ReadMeasure[elt].round(5)
             print(elt + " '''''''''''''''''''''''''''''''''''  is  now  good!")

    ReadMeasure.round(4)
    #print(ReadMeasure.loc[[nanelt],[elt]])
                #ReadMeasure[elt] = ReadMeasure[elt].str.replace(',','.').astype(float)


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
    print("########################################__ Anfang der Wiederersellung___################################")
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

def writePerformanceModell (ModellParmeter):
    #df = pd.DataFrame.from_dict(ModellParmeter, orient='index')
    #df.columns = ModellParmeter.
    Perform = pd.DataFrame(ModellParmeter)
    print(Perform)

create_test_train_data()