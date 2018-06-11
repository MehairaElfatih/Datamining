import pandas as pd
from os import path
import time
import numpy as np
from matplotlib import pyplot as plt

class Explorative:
    def __init__(self, dataset):
        self.data = dataset

    def transformation (self):
        self.data['P-Geschlecht'] = self.data['P-Geschlecht'].str.replace('1', 'm')
        self.data['P-Geschlecht'] = self.data['P-Geschlecht'].str.replace('0', 'w')

        self.data['P-Altersklasse'] = self.data['P-Altersklasse'].str.replace('0', '20-29')
        self.data['P-Altersklasse'] = self.data['P-Altersklasse'].str.replace('1', '30-39')
        self.data['P-Altersklasse'] = self.data['P-Altersklasse'].str.replace('2', '40-49')
        self.data['P-Altersklasse'] = self.data['P-Altersklasse'].str.replace('3', '50-59')
        self.data['P-Altersklasse'] = self.data['P-Altersklasse'].str.replace('4', '60-69')

        return self.data

    # Vergleich Klassenverteilung in Trainings- und +Testmenge (Klassenhistogramme)
    def klassenverteilung (self):
        #self.data.plot('P-Altersklasse', kind = 'bar')

        N = 5
        menMeans = (20, 35, 30, 35, 27)
        womenMeans = (25, 32, 34, 20, 25)
        menStd = (2, 3, 4, 1, 2)
        womenStd = (3, 5, 2, 3, 3)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, menMeans, width, yerr=menStd)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans, yerr=womenStd)

        plt.ylabel('Scores')
        plt.title('Scores by group and gender')
        plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
        plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('Men', 'Women'))

        plt.show()

    # Von welchen Merkmalen hängt die Klasse stark ab (Korrelation)?

    def korrelation(self):
        self.data.corr()
        plt.matshow(self.data.corr())
        plt.show()
        # Boxplot wichtiger Merkmale über der Klasse

    # Bivariate Abhängigkeiten: Korrelationsmatrix der Merkmale, stärkste Zusammenhänge, Streudiagramm Merkmal1 vs. Merkmal2

    def abhangigkeit(self):
        pass


path_orignalFiles = path.curdir + '/../../Daten/original/'
ReadMeasure = pd.read_csv(path_orignalFiles + 'measures.csv', sep=';', decimal=',')
test = Explorative(ReadMeasure)
test.klassenverteilung()
