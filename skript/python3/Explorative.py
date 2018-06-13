import pandas as pd
from os import path
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from matplotlib import cm as cm

path_final = path.curdir + '/../../Abbildungen/final/'
path_exploration = path.curdir + '/../../Abbildungen/exploration/'
path_erweiter = path.curdir + '/../../Daten/erweitert/'
path_ergebnis = path.curdir + '/../../Daten/ergebnis/'


class Explorative:
    def __init__(self):
        self.ReadMeasure = pd.read_csv(path_erweiter + 'Measure_Umgewandelt.csv', sep=';', decimal=',')
        self.ReadTrain = pd.read_csv(path_erweiter + 'TrainMeasure.csv', sep=';', decimal=',')
        self.ReadTest = pd.read_csv(path_erweiter + 'TestMeasure.csv', sep=';', decimal=',')
        self.Readperformance = pd.read_csv(path_ergebnis + 'Performance.csv', sep=';', decimal='.')

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
        print("romeo")
        plt.matshow(df.corr)
        plt.show()

    # Von welchen Merkmalen hängt die Klasse stark ab (Korrelation)?
    def korrelation(self, df):
        """
        fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(test.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
        :param df:
        :return:
        """
        df=df.drop(['P-KennungAnonym'], axis=1)
        print("korrelation")
        correlation = df.corr()
        figure = plt.figure()
        ax = figure.add_subplot(111) #gr0ße des Plotsbildschirm
        cmap = cm.get_cmap('jet',30)
        cax = ax.imshow(correlation, interpolation="nearest", cmap=cmap)
        figure.colorbar(cax)
        ticks = [1,9,0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        #print(df.columns.values)
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
        X = df.loc[:, df.columns != 'P-Altersklasse'].values
        y = df['P-Altersklasse'].values
        print(X.shape)
        X_new = SelectKBest(f_classif, k=94).fit_transform(X,y)
        print(type(X_new))
        plt.plot(X_new[0], X_new[1], 'bo')
        plt.savefig(path_exploration + 'wichtige_merkmale' + '.pdf')

    def Plot_perfromance (self,df):
        perform_svc = df.loc[df['name'] =='SVC',]
        perform_rf = df.loc[df['name'] == 'Random Forest',] #MLPClassifier ,DecisionTreeClassifier  ,KNeighborsClassification
        perform_knn = df.loc[df['name'] == 'KNeighborsClassification',]
        perform_mlpc = df.loc[df['name'] == 'MLPClassifier',]
        perform_dtc = df.loc[df['name'] == 'DecisionTreeClassifier']
        #print(perform_dtc)

        fig = plt.figure()
        plt.grid(color='gray', linestyle='-', linewidth=0.1)
        plt.xlabel('accuracy')
        plt.ylabel('Testfehler')
        
        plt.title('performance of SVC')
        plt.scatter(perform_svc['Acuracy'], perform_svc['TestFehle'], marker='X')        
        plt.savefig(path_exploration + 'SVC' + '.pdf')
        
        plt.title('performance of Random Forest')
        plt.scatter(perform_rf['Acuracy'], perform_rf['TestFehle'], marker='X')        
        plt.savefig(path_exploration + 'Random Forest' + '.pdf')
        
        plt.title('performance of KNeighborsClassification')
        plt.scatter(perform_knn['Acuracy'], perform_knn['TestFehle'], marker='X')        
        plt.savefig(path_exploration + 'KNeighborsClassification' + '.pdf')
        
        plt.title('performance of MLPClassifier')
        plt.scatter(perform_mlpc['Acuracy'], perform_mlpc['TestFehle'], marker='X')        
        plt.savefig(path_exploration + 'MLPClassifier' + '.pdf')
        
        plt.title('performance of DecisionTreeClassifier')
        plt.scatter(perform_dtc['Acuracy'], perform_dtc['TestFehle'], marker='X')        
        plt.savefig(path_exploration + 'DecisionTreeClassifier' + '.pdf')

    def konfusion_Marix(self,confisionMatrice, accuracy, name):
        sns_plot = sns.heatmap(confisionMatrice, annot=True, fmt="d", cbar=False)
        sns_plot.set_title('accuracy = ' + str(accuracy))
        sns_plot.figure.savefig(path_final + name)

    def data_normalisation (self, df):
        df_norm = (df.iloc[:,:94] - df.iloc[:,:94].mean())/(df.iloc[:,:94].max() - df.iloc[:,:94].min())
        df.iloc[:,:94] =df_norm
        return df

