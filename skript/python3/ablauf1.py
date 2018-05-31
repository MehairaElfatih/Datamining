from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import Kreuzvalidierung as kv
from os import path

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
weights_knn = 'uniform'  # Moegliche Werte: 'uniform','distance'
prozessor_kerne_knn = -1  # -1 Bedeutet, dass die maximale Anzahl an CPU-Kernen benutzt wird
algorithm = 'ball_tree' #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional

#___________________________________________________________________________________________________________________
kflod = 10
ausgewahltet_Modell = 1
ModellParam = {}

if (ausgewahltet_Modell == 1):
    modell = SVC(C=C,kernel=kernel,degree=3,gamma=gamma,class_weight=class_weight,coef0=coef0,max_iter=max_iter)
if (ausgewahltet_Modell == 2):
    modell = KNeighborsClassifier(n_neighbors= n_neighbors_knn,weights=weights_knn,n_jobs=prozessor_kerne_knn, algorithm=algorithm)
    ModellParam['Name'] = 'KNeighborsClassification'
    ModellParam['n_neighbors'] = n_neighbors_knn
    ModellParam['weights_knn'] = weights_knn
    ModellParam['prozessor_kerne_knn'] = prozessor_kerne_knn
    ModellParam['algorithm'] = algorithm

cv= kv.validation(modell,kflod)
dauert, scoresMean =cv.kreuzvalidierng_model()
print("Accuracy: %0.2f : testFehler: %0.2f , in %0.5f" % (np.mean(scoresMean), 1-np.mean(scoresMean), dauert))
print(ModellParam)
