from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import Kreuzvalidierung as kv


"""
Parameter fuer jedes Modell Eingeben Und dann Modell Trainieren 
"""

#_SVC___________________________________________________________________________
# modell 1
C=1.0                   # der Parameter der  Sanktion vom dem Fehler (1.0 ist der max  value)
class_weight=0.1      #
coef0=0.0               # nur für poly and sigmoid kernel nötig
degree=3                # nur bei Polynomial kernel anwendbar
gamma='auto'            # Kernel Coeffient (rbf,poly, and sigmoid)
kernel='poly'            #
max_iter=-1             #

# KNeighborsRegressor______ Modell 2 __________________________
n_neighbors_knn = 5 # Anzahl Nachbarn
weights_knn = 'distance'  # Moegliche Werte: 'uniform','distance'
prozessor_kerne_knn = -1  # -1 Bedeutet, dass die maximale Anzahl an CPU-Kernen benutzt wird
algorithm = 'auto' #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional

# ____ Decision Tree Classificator_____________________________
#modell 3
criterion ='gini' # #criterion : string, optional (default=”gini”  impurity and “entropy”)
splitter = 'best'#splitter : string, optional (default=”best” random)
max_depth = 9 #max_depth : int or None, optional (default=None)

#___________________________________________________________________________________________________________________
kflod = 10
ausgewahltet_Modell = 3

if (ausgewahltet_Modell == 1):
    ModellName = 'SVC'
    modell = SVC(C=C,kernel=kernel,degree=degree,gamma=gamma,class_weight=class_weight,coef0=coef0,max_iter=max_iter)
    Parameter ='C:'+ str(C) + 'kenel:' + str(kernel) + ' degree:' + str(degree)+' gamma :' + str(gamma) + 'class_weight:' + str(class_weight) + ' coef0:' + str(coef0)\
        + ' max_iter :' + str(max_iter)
if (ausgewahltet_Modell == 2):
    modell = KNeighborsClassifier(n_neighbors= n_neighbors_knn,weights=weights_knn,n_jobs=prozessor_kerne_knn, algorithm=algorithm)
    ModellName = 'KNeighborsClassification'
    Parameter = ' n_neighbors:'+ str(n_neighbors_knn) + ',weights:'+ str(weights_knn) + ',n_jobs:'+ str(prozessor_kerne_knn) + ',algorithm:'+ algorithm
if (ausgewahltet_Modell == 3):
    ModellName = 'DecisionTreeClassifier'
    modell = DecisionTreeClassifier(criterion= criterion,splitter=splitter,max_depth=max_depth)
    Parameter = 'criterion:' + str(criterion) + ',splitter' + str(splitter) + ',max_depth' + str(max_depth)

cv= kv.validation(modell,kflod)
dauert, scoresMean =cv.kreuzvalidierng_model()

ModellParamdict = {'name': ModellName, 'Acuracy': np.mean(scoresMean) , 'TestFehle': 1-np.mean(scoresMean),'Parameter': Parameter }


cv.writePerformanceModell(ModellParamdict)
