from sklearn.svm import SVC
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
kernel='rbf'            #
max_iter=-1             #

#___________________________________________________________________________________________________________________
kflod = 10
ausgewahltet_Modell = 1

if (ausgewahltet_Modell == 1):
    modell = SVC(C=C,kernel=kernel,degree=3,gamma=gamma,class_weight=class_weight,coef0=coef0,max_iter=max_iter)


print(path.abspath(path.curdir + '/../../Daten/erweitert/'))
cv= kv.validation(modell,kflod)
cv.kreuzvalidierng_model()