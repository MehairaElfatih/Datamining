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
        self.Readperformance90 = pd.read_csv(path_ergebnis + 'performance_90.csv', sep=';', decimal='.')

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
        measures_head = list(df.head(0))
        sns.set_style("whitegrid")
        y_axes = "P-Altersklasse"
        for x in measures_head[0::4]:
            index = measures_head.index(x)
            if (index == 92):
                ax1 = plt.subplot(221)
                sns.boxplot(x=x, y=y_axes, data=df, orient="h", palette="Set2")

                ax2 = plt.subplot(222, sharey=ax1)
                sns.boxplot(x=measures_head[index + 1], y=y_axes, data=df, orient="h", palette="Set2")

                ax3 = plt.subplot(223, sharey=ax1)
                sns.boxplot(x=measures_head[index + 2], y=y_axes, data=df, orient="h", palette="Set2")
                plt.savefig(path_exploration + 'BoxPlt_wichtigermerkmal' + str(x) + '.png')
                plt.show()
            else:
                ax1 = plt.subplot(221)
                sns.boxplot(x=x, y=y_axes, data=df, orient="h", palette="Set2")

                ax2 = plt.subplot(222, sharey=ax1)
                sns.boxplot(x=measures_head[index + 1], y=y_axes, data=df, orient="h", palette="Set2")

                ax3 = plt.subplot(223, sharey=ax1)
                sns.boxplot(x=measures_head[index + 2], y=y_axes, data=df, orient="h", palette="Set2")

                ax3 = plt.subplot(224, sharey=ax1)
                sns.boxplot(x=measures_head[index + 3], y=y_axes, data=df, orient="h", palette="Set2")
                plt.savefig(path_exploration + 'BoxPlt_wichtigermerkmal' + str(x) + '.png')
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
        
        """

        # calculate the correlation matrix
        df = df.drop(['P-KennungAnonym'], axis=1)
        print("korrelation")
        correlation = df.corr()

        #########plot correlation
        def magnify():
            return [dict(selector="th",
                         props=[("font-size", "7pt")]),
                    dict(selector="td",
                         props=[('padding', "0em 0em")]),
                    dict(selector="th:hover",
                         props=[("font-size", "8pt")]),
                    dict(selector="tr:hover td:hover",
                         props=[('max-width', '200px'),
                                ('font-size', '8pt')])
                    ]

            correlation.style.background_gradient(cmap, axis=1) \
            .set_properties(**{'max-width': '30px', 'font-size': '8pt'}) \
            .set_caption("Hover to magify") \
            .set_precision(2) \
            .set_table_styles(magnify())

        # plot the heatmap
        plt.figure(num=None, figsize=(30, 30), dpi=80, facecolor='w', edgecolor='k')
        sns.heatmap(correlation,
                    xticklabels=correlation.columns,
                    yticklabels=correlation.columns)
        plt.savefig(path_exploration + 'Korrelation_ matshow' + '.pdf')
        plt.show()




    # Bivariate Abhängigkeiten: Korrelationsmatrix der Merkmale, stärkste Zusammenhänge, Streudiagramm Merkmal1 vs. Merkmal2
    def wichtiger_merkmale(self,df):
        # # Boxplot wichtiger Merkmale über der Klasse
        # # http://scikit-learn.org/stable/modules/feature_selection.html
        # X = df.loc[:, df.columns != 'P-Altersklasse'].values
        # y = df['P-Altersklasse'].values
        # print(X.shape)
        # X_new = SelectKBest(f_classif, k=94).fit_transform(X,y)
        # print(type(X_new))
        # plt.yticks(y)
        # plt.plot(X_new[0], X_new[1], 'bo')
        # plt.savefig(path_exploration + 'wichtige_merkmale' + '.pdf')
        correlationhight = {}
        measures_head = list(df.head(0))
        for n in measures_head:
            correlationhight[n] = df[n].corr(df['P-Altersklasse'])
            if (-0.2 < df[n].corr(df['P-Altersklasse']) < 0.2):
                pass
                #print("Low correlation")
                #print(n + " " + str(df[n].corr(df['P-Altersklasse'])))
            elif (-1.0 < df[n].corr(df['P-Altersklasse']) < -0.4 or 0.4 < df[n].corr(
                    df['P-Altersklasse']) < 1.0):
                pass
        dfcorrelat = pd.DataFrame.from_dict(correlationhight,orient='index')

        plt.bar(range(len(correlationhight)), list(correlationhight.values()), align='center')
        plt.xticks(range(len(correlationhight)), list(correlationhight.keys()))
        #plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.savefig(path_exploration + 'Korrelation_Klasse_Merkmal' + '.png')
        plt.show()
        print(dfcorrelat)

    def Plot_perfromance (self,df):
        perform_svc = df.loc[df['name'] =='SVC',]
        perform_rf = df.loc[df['name'] == 'Random Forest',] #MLPClassifier ,DecisionTreeClassifier  ,KNeighborsClassification
        perform_knn = df.loc[df['name'] == 'KNeighborsClassification',]
        perform_mlpc = df.loc[df['name'] == 'MLPClassifier',]
        perform_dtc = df.loc[df['name'] == 'DecisionTreeClassifier']

        print(perform_dtc)

        fig = plt.figure()
        plt.grid(color='gray', linestyle='-', linewidth=0.1)
        plt.xlabel('Accuracy')
        plt.ylabel('Testfehler')
        
        plt.title('performance of SVC')
        plt.scatter(perform_svc['Accuracy'], perform_svc['TestFehle'], marker='X')
        #plt.savefig(path_exploration + 'SVC' + '.pdf')
        
        plt.title('performance of Random Forest')
        plt.scatter(perform_rf['Accuracy'], perform_rf['TestFehle'], marker='X')
        #plt.savefig(path_exploration + 'Random Forest' + '.pdf')
        
        plt.title('performance of KNeighborsClassification')
        plt.scatter(perform_knn['Accuracy'], perform_knn['TestFehle'], marker='X')
        #plt.savefig(path_exploration + 'KNeighborsClassification' + '.pdf')
        
        plt.title('performance of MLPClassifier')
        plt.scatter(perform_mlpc['Accuracy'], perform_mlpc['TestFehle'], marker='X')
        #plt.savefig(path_exploration + 'MLPClassifier' + '.pdf')
        
        plt.title('performance of DecisionTreeClassifier')
        plt.scatter(perform_dtc['Accuracy'], perform_dtc['TestFehle'], marker='X')
        plt.savefig(path_exploration + 'DecisionTreeClassifier' + '.pdf')

    def Plot_perfromance90(self, df):
        perform_rf = df.loc[df['name'] == 'Random Forest',]  # MLPClassifier ,DecisionTreeClassifier  ,KNeighborsClassification

        #fig = plt.figure()
        plt.scatter(perform_rf['Accuracy'], perform_rf['TestFehle'], marker='X')
        plt.xlabel('Accuracy')
        plt.ylabel('Testfehler')

        plt.title('performance of Random Forest')
        plt.scatter(perform_rf['Accuracy'], perform_rf['TestFehle'], marker='X')
        plt.savefig(path_exploration + 'Random Forest90' + '.pdf')


    def konfusion_Marix(self,confisionMatrice, accuracy, name):
        sns_plot = sns.heatmap(confisionMatrice, annot=True, fmt="d", cbar=False)
        sns_plot.set_title('accuracy = ' + str(accuracy))
        sns_plot.figure.savefig(path_final + name)

    def data_normalisation (self, df):
        df_norm = (df.iloc[:,:94] - df.iloc[:,:94].mean())/(df.iloc[:,:94].max() - df.iloc[:,:94].min())
        df.iloc[:,:94] =df_norm
        return df

