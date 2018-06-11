from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import Kreuzvalidierung as kv


"""
Parameter fuer jedes Modell Eingeben Und dann Modell Trainieren 
"""

#_SVC___________________________________________________________________________
# modell 1
C=1.0                   # der Parameter der  Sanktion vom dem Fehler (1.0 ist der max  value)
#class_weight=0     #
coef0=0.0               # nur für poly and sigmoid kernel nötig
degree=3                # nur bei Polynomial kernel anwendbar
gamma='auto'            # Kernel Coeffient (rbf,poly, and sigmoid) rbf und Signoid sind schlescht
kernel='linear'            #
max_iter=-1             #

# KNeighborsRegressor______ Modell 2 __________________________
n_neighbors_knn = 2 # Anzahl Nachbarn
weights_knn = 'distance'  # Moegliche Werte: 'uniform','distance'
prozessor_kerne_knn = -1  # -1 Bedeutet, dass die maximale Anzahl an CPU-Kernen benutzt wird
algorithm = 'ball_tree' #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional

# ____ Decision Tree Classificator_____________________________
#modell 3
criterion ='entropy' # #criterion : string, optional (default=”gini”  impurity and “entropy”)
splitter = 'random'#splitter : string, optional (default=”best” random)
max_depth = 5 #max_depth : int or None, optional (default=None)

#_______________MlPClassifier____________________________________________________________________
hidden_layer_sizes = (100,) #hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
activation = 'relu'  #activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
solver = 'sgd' #solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
alpha = 0.00001
learnin_rate = 'adaptive'  # learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
momentum = 0.5 #momentum : float, default 0.9 [0,1]
#
#___________________________________________________________________________________________________________________
kflod = 5
ausgewahltet_Modell = 4

if (ausgewahltet_Modell == 1):
    ModellName = 'SVC'
    modell = SVC(C=C,kernel=kernel,degree=degree,gamma=gamma,coef0=coef0,max_iter=max_iter)
    Parameter ='C ='+ str(C) + 'kenel =' + str(kernel) + ' degree =' + str(degree)+' gamma =' + str(gamma) + 'class_weight ='  + ' coef0 =' + str(coef0)\
        + ' max_iter :' + str(max_iter)
if (ausgewahltet_Modell == 2):
    modell = KNeighborsClassifier(n_neighbors= n_neighbors_knn,weights=weights_knn,n_jobs=prozessor_kerne_knn, algorithm=algorithm)
    ModellName = 'KNeighborsClassification'
    Parameter = ' n_neighbors ='+ str(n_neighbors_knn) + ',weights ='+ str(weights_knn) + ',n_jobs ='+ str(prozessor_kerne_knn) + ',algorithm ='+ algorithm
if (ausgewahltet_Modell == 3):
    ModellName = 'DecisionTreeClassifier'
    modell = DecisionTreeClassifier(criterion= criterion,splitter=splitter,max_depth=max_depth)
    Parameter = 'criterion =' + str(criterion) + ',splitter =' + str(splitter) + ',max_depth =' + str(max_depth)
if(ausgewahltet_Modell == 4):
    ModellName = 'MLPClassifier'
    modell = MLPClassifier(hidden_layer_sizes= hidden_layer_sizes, activation = activation, solver=solver, alpha=alpha, learning_rate=learnin_rate, momentum=momentum)
    Parameter = 'hidden_layer_sizes =' + str(hidden_layer_sizes) + 'activation = ' + activation + 'solver = '+ solver + ' alpha =' + str(alpha) + 'learning_rate=' + learnin_rate + 'momentum = ' + str(momentum)
cv= kv.validation(modell,kflod)
dauert, scoresMean =cv.kreuzvalidierng_model()

ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean) , 'TestFehle': 1-np.mean(scoresMean),'Parameter': Parameter }


cv.writePerformanceModell(ModellParamdict)

#cv.generated_model()
