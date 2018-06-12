import pandas as pd
from os import path
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif


path_exploration = path.curdir + '/../../Abbildungen/exploration/'
path_erweiter = path.curdir + '/../../Daten/erweitert/'


class Explorative:
    def __init__(self):
        self.ReadMeasure = pd.read_csv(path_erweiter + 'Measure_Umgewandelt.csv', sep=';', decimal=',')
        self.ReadTrain = pd.read_csv(path_erweiter + 'TrainMeasure.csv', sep=';', decimal=',')
        self.ReadTest = pd.read_csv(path_erweiter + 'TestMeasure.csv', sep=';', decimal=',')

    def transformation(self, data):
        data['P-Geschlecht'] = data['P-Geschlecht'].replace(1, 'm')
        data['P-Geschlecht'] = data['P-Geschlecht'].replace(0, 'w')

        data['P-Altersklasse'] = data['P-Altersklasse'].replace(0, '20-29')
        data['P-Altersklasse'] = data['P-Altersklasse'].replace(1, '30-39')
        data['P-Altersklasse'] = data['P-Altersklasse'].replace(2, '40-49')
        data['P-Altersklasse'] = data['P-Altersklasse'].replace(3, '50-59')
        data['P-Altersklasse'] = data['P-Altersklasse'].replace(4, '60-69')

        return data

    # Vergleich Klassenverteilung in Trainings- und +Testmenge (Klassenhistogramme)
    def klassenverteilungTrain(self):
        self.ReadTrain = self.transformation(self.ReadTrain)

        plt.title('Klassenverteilung train')
        plt.xlabel('Alter Klasse')
        plt.ylabel("Anzahl")
        plt.hist(self.ReadTrain['P-Altersklasse'])
        titletrain = 'Klassenverteilung train'
        plt.savefig(path_exploration + titletrain + '.pdf')

    def klassenverteilungTest(self):
        self.ReadTest = self.transformation(self.ReadTest)
        titletest = 'Klassenverteilung test'
        plt.title('Klassenverteilung test')
        plt.xlabel('Alter Klasse')
        plt.ylabel("Anzahl")
        plt.hist(self.ReadTest['P-Altersklasse'])
        plt.savefig(path_exploration + titletest + '.pdf')

    def boxsplot(self, df):
        # df = self.transformation(df)
        # bplot = sns.boxplot(x =['P-Geschlecht'],y = ['P-Altersklasse'],data = df.values,width = 0.5, palette = "colorblind" )
        df.boxplot(by=['P-Altersklasse'], column=['P-Geschlecht'], rot=0)

    # Von welchen Merkmalen hängt die Klasse stark ab (Korrelation)?
    def korrelation(self, df):
        df=df.drop(['P-KennungAnonym'], axis=1)
        print("korrelation")
        correlation = df.corr()
        figure = plt.figure()
        ax = figure.add_subplot(111) #gr0ße des Plotsbildschirm
        cax = ax.matshow(correlation, vmin=-1, vmax=1)
        figure.colorbar(cax)
        ticks = np.arange(0, 9, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(df.columns.values)
        ax.set_yticklabels(df.columns.values)
        #plt.savefig(path_exploration + 'Korrelation_ matshow' + '.pdf')
        plt.show()
        
        #scatter_matrix(df)
        #plt.savefig(path_exploration + 'Korrelation_ matrix' + '.pdf')
        #plt.show() 
    # Bivariate Abhängigkeiten: Korrelationsmatrix der Merkmale, stärkste Zusammenhänge, Streudiagramm Merkmal1 vs. Merkmal2
    def wichtiger_merkmale(self,df):
        # Boxplot wichtiger Merkmale über der Klasse
        # http://scikit-learn.org/stable/modules/feature_selection.html
        #X = df.loc[:, df.columns != 'P-Altersklasse'].values
        X = df.values
        y = df['P-Altersklasse'].values
        print(X.shape)
        X_new = SelectKBest(f_classif, k=94).fit_transform(X,y)
        print(type(X_new))
        #bplot = sns.boxplot(x= X_new[0], y=X_new[0], data=X_new, width=0.5, palette="colorblind")
        plt.plot(X_new[0], X_new[1], 'bo')
        #plt.show()
        plt.savefig(path_exploration + 'wichtige_merkmale' + '.pdf')
        


#ReadTrain = pd.read_csv(path_erweiter + 'Measure_Umgewandelt.csv', sep=';', decimal=',')
#ReadTest = pd.read_csv(path_erweiter + 'TestMeasure.csv', sep=';', decimal=',')
test = Explorative()
# test.klassenverteilungTrain()
# test.klassenverteilungTest()
readTest = test.ReadMeasure
test.wichtiger_merkmale(readTest)
