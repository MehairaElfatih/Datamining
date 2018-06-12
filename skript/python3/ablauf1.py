from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import Kreuzvalidierung as kv
import Utility as uty

#___________________________________________________________________________________________________________________
kflod = 5

for modell in range(5,6):
    ausgewahltet_Modell = modell

    if (ausgewahltet_Modell == 1):
        # _SVC___________________________________________________________________________
        C = 1.0  # der Parameter der  Sanktion vom dem Fehler (1.0 ist der max  value)
        degree = [3]  # nur bei Polynomial kernel anwendbar
        gamma = 'auto'  # float, optional (default=’auto’)
        coef0 = 0.0
        kernel = [ 'poly']  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid  ['linear', 'poly', 'rbf', 'sigmoid']

        ModellName = 'SVC'
        for kern in kernel:
            for dg in degree:
                print(dg)
                for rd_st in range(1,10):
                        print(kern)
                        modell = SVC(C=C, kernel=kern, gamma=gamma, degree= dg,coef0=coef0, random_state =rd_st,max_iter=-1)
                        Parameter = 'C =' + str(C) + '  kenel =' + str(kern) + '  gamma =' + str(
                            gamma)  + '  coef0 =' + str(coef0) \
                                    + '   random_state :' + str(rd_st) + '  max_iter :' + str(-1) + ' degree = ' + str(dg)

                        cv = kv.validation(modell, kflod)
                        dauert, scoresMean = cv.kreuzvalidierng_model()

                        ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean),
                                           'TestFehle': 1 - np.mean(scoresMean), 'Parameter': Parameter}

                        perf = uty.Utility()
                        perf.writePerformanceModell(ModellParamdict)

    if (ausgewahltet_Modell == 2):
        # KNeighborsRegressor______ Modell 2 __________________________
        n_neighbors_knn = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Anzahl Nachbarn
        weights_knn = ['uniform', 'distance']  # Moegliche Werte: 'uniform','distance'
        prozessor_kerne_knn = -1  # -1 Bedeutet, dass die maximale Anzahl an CPU-Kernen benutzt wird
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']  # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
        ModellName = 'KNeighborsClassification'
        for neigbor in n_neighbors_knn:
            for weight in weights_knn:
                for alg in algorithm:
                    modell = KNeighborsClassifier(n_neighbors=neigbor, weights=weight, n_jobs=prozessor_kerne_knn,
                                                  algorithm=alg)
                    Parameter = ' n_neighbors =' + str(neigbor) + ',weights =' + str(weight) + ',n_jobs =' + str(
                        prozessor_kerne_knn) + ',algorithm =' + alg

                    cv = kv.validation(modell, kflod)
                    dauert, scoresMean = cv.kreuzvalidierng_model()

                    ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean),
                                       'TestFehle': 1 - np.mean(scoresMean), 'Parameter': Parameter}

                    perf = uty.Utility()
                    perf.writePerformanceModell(ModellParamdict)
    if (ausgewahltet_Modell == 3):
        # ____ Decision Tree Classificator_____________________________
        # modell 3
        ModellName = 'DecisionTreeClassifier'
        criterion = ['entropy', 'gini']  # #criterion : string, optional (default=”gini”  impurity and “entropy”)
        splitter = ['random', 'best', ]  # splitter : string, optional (default=”best” random)
        max_depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                     30]  # max_depth : int or None, optional (default=None)
        for deph in max_depth:
            for crit in criterion:
                for split in splitter:
                    modell = DecisionTreeClassifier(criterion=crit, splitter=split, max_depth=deph)
                    Parameter = 'criterion =' + str(crit) + ',splitter =' + str(split) + ',max_depth =' + str(
                        deph)

                    cv = kv.validation(modell, kflod)
                    dauert, scoresMean = cv.kreuzvalidierng_model()

                    ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean),
                                       'TestFehle': 1 - np.mean(scoresMean), 'Parameter': Parameter}

                    perf = uty.Utility()
                    perf.writePerformanceModell(ModellParamdict)

    if (ausgewahltet_Modell == 4):
        # _______________MlPClassifier____________________________________________________________________
        #hidden_layer_sizes = [(70,),(75,)(80,),(85,),(90,),(100,)]  # hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        activation = 'relu'  # activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
        solver = ['sgd', 'lbfgs', 'adam']  # solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
        alpha = [0.00001]
        learnin_rate = ['adaptive', 'constant',
                        'invscaling']  # learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
        momentum = [0.9]  # momentum : float, default 0.9 [0,1]
        ModellName = 'MLPClassifier'

        for i in range(96,100):
            for sl in solver:
                for al in alpha:
                    for mom in momentum:
                        for learnin in learnin_rate:
                            modell = MLPClassifier(hidden_layer_sizes=(i,), activation=activation, solver=sl, alpha=al,
                                                   learning_rate=learnin, momentum=mom)
                            Parameter = 'hidden_layer_sizes =' + str(
                                (i,)) + 'activation = ' + activation + 'solver = ' + sl + ' alpha =' + str(
                                al) + 'learning_rate=' + learnin + 'momentum = ' + str(mom)

                            cv = kv.validation(modell, kflod)
                            dauert, scoresMean = cv.kreuzvalidierng_model()

                            ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean),
                                               'TestFehle': 1 - np.mean(scoresMean), 'Parameter': Parameter}
                            perf = uty.Utility()
                            perf.writePerformanceModell(ModellParamdict)
    if (ausgewahltet_Modell == 5):
        # modell 5
        ModellName = 'Random Forest'
        criterion = ['entropy', 'gini']  # #criterion : string, optional (default=”gini”  impurity and “entropy”)
        splitter = ['random', 'best', ]  # splitter : string, optional (default=”best” random)
        # max_depth : int or None, optional (default=None)
        for deph in range(5, 30):
            for crit in criterion:
                for estimator in range(1, 30):
                    modell = RandomForestClassifier(n_estimators=estimator, criterion=crit, max_depth=deph,
                                                    max_features='auto')
                    Parameter = 'n_estimators = ' + str(estimator) + ' criterion =' + str(crit) + ',max_depth =' + str(
                        deph) + 'max_features =' + 'auo'

                    cv = kv.validation(modell, kflod)
                    dauert, scoresMean = cv.kreuzvalidierng_model()

                    ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean),
                                       'TestFehle': 1 - np.mean(scoresMean), 'Parameter': Parameter}

                    perf = uty.Utility()
                    perf.writePerformanceModell(ModellParamdict)

#cv.generated_model()
