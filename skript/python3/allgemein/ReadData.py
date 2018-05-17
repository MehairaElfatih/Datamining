from builtins import print

import pandas as pd
import csv

#read files
from numpy.ma.extras import column_stack
from sphinx.testing.path import path

path_orignalFiles = '../../../Daten/original/'
path_erweiter  = '../../../Daten/erweitert/'

ReadMeasure = pd.read_csv(path_orignalFiles +'measures.csv', sep=';')
ReadToPedict = pd.read_csv(path_orignalFiles +'to_predict.csv', sep=';')
ReadTrainMeasure = pd.read_csv(path_erweiter +'TrainMeasure.csv', sep=';')
ReadTestMeasure = pd.read_csv(path_erweiter +'TrainMeasure.csv', sep=';')

# Split data Test und Train
def create_test_train_data():
    testMeasure = pd.DataFrame()
    trainMeasure = pd.DataFrame()
    IdTestPerson =  pd.Series([3, 26, 27, 32, 37, 44, 47, 51, 54, 57, 60, 61, 63, 66, 70, 90, 95, 109, 111, 116, 120, 123, 134, 149, 151])
    dfreadmeasure = readmeasure.rename(columns={'P-KennungAnonym': 'Pid'})
    dfreadmeasure = readmeasure.rename(columns={'L-firstStep': 'LfirstStep'})
    cols = list(dfreadmeasure.columns)
    for id in IdTestPerson:
        testMeasure.merge(dfreadMeasure.loc[dfreadMeasure['Pid'] == id], on= ['LfirstStep', 'L-lastStep', 'L-meanAmplMidSwing', 'L-amplDiffTCIC', 'L-meanStride','L-meanStance'
                          , 'L-meanSwing', 'L-SwingStride', 'L-StanceStride', 'L-meanOneLegStance', 'L-meanDoubleLegSupport', 'L-oneLegStanceStride', 'L-doubleLegSupportStride'
                          , 'L-weg', 'L-meanStrideLen', 'L-meanStepHeight', 'L-aufsetzwinkel', 'L-abrollwinkel', 'L-velocity', 'L-velocity2', 'L-cadence', 'L-OKtilt', 'L-ROMShoulderX'
                          , 'L-ROMShoulderY', 'L-ROMShoulderZ', 'L-meanROMHipX', 'L-meanROMSpineX', 'L-meanROMHipY', 'L-meanROMSpineY', 'L-meanROMHipZ', 'L-meanROMSpineZ', 'L-StrideLength'
                          , 'L-StrideLength2', 'L-StepLength', 'R-firstStep', 'R-lastStep', 'R-meanAmplMidSwing', 'R-amplDiffTCIC', 'R-meanStride', 'R-meanStance', 'R-meanSwing'
                          , 'R-SwingStride', 'R-StanceStride', 'R-meanOneLegStance', 'R-meanDoubleLegSupport', 'R-oneLegStanceStride', 'R-doubleLegSupportStride', 'R-weg'
                          , 'R-meanStrideLen', 'R-meanStepHeight', 'R-aufsetzwinkel', 'R-abrollwinkel', 'R-velocity', 'R-velocity2', 'R-cadence', 'R-OKtilt', 'R-ROMShoulderX'
                          , 'R-ROMShoulderY', 'R-ROMShoulderZ', 'R-meanROMHipX', 'R-meanROMSpineX', 'R-meanROMHipY', 'R-meanROMSpineY', 'R-meanROMHipZ', 'R-meanROMSpineZ', 'R-StrideLength'
                          , 'R-StrideLength2', 'R-StepLength', 'S-lastStep', 'S-meanAmplMidSwing', 'S-amplDiffTCIC', 'S-meanStride', 'S-meanStance', 'S-meanSwing', 'S-SwingStride', 'S-StanceStride'
                          , 'S-meanOneLegStance', 'S-meanDoubleLegSupport', 'S-oneLegStanceStride', 'S-doubleLegSupportStride', 'S-weg', 'S-meanStrideLen', 'S-meanStepHeight', 'S-aufsetzwinkel'
                          , 'S-abrollwinkel', 'S-cadence', 'S-meanROMHipX', 'S-meanROMSpineX', 'S-meanROMHipY', 'S-meanROMSpineY', 'S-meanROMHipZ', 'S-meanROMSpineZ', 'S-StrideLength2', 'S-StepLength'
                          , 'P-Altersklasse', 'P-Geschlecht', 'Pid'])
    for id in IdTestPerson:
        trainMeasure.merge(dfreadMeasure[dfreadMeasure.Pid != id], on= ['LfirstStep', 'L-lastStep', 'L-meanAmplMidSwing', 'L-amplDiffTCIC', 'L-meanStride','L-meanStance'
                          , 'L-meanSwing', 'L-SwingStride', 'L-StanceStride', 'L-meanOneLegStance', 'L-meanDoubleLegSupport', 'L-oneLegStanceStride', 'L-doubleLegSupportStride'
                          , 'L-weg', 'L-meanStrideLen', 'L-meanStepHeight', 'L-aufsetzwinkel', 'L-abrollwinkel', 'L-velocity', 'L-velocity2', 'L-cadence', 'L-OKtilt', 'L-ROMShoulderX'
                          , 'L-ROMShoulderY', 'L-ROMShoulderZ', 'L-meanROMHipX', 'L-meanROMSpineX', 'L-meanROMHipY', 'L-meanROMSpineY', 'L-meanROMHipZ', 'L-meanROMSpineZ', 'L-StrideLength'
                          , 'L-StrideLength2', 'L-StepLength', 'R-firstStep', 'R-lastStep', 'R-meanAmplMidSwing', 'R-amplDiffTCIC', 'R-meanStride', 'R-meanStance', 'R-meanSwing'
                          , 'R-SwingStride', 'R-StanceStride', 'R-meanOneLegStance', 'R-meanDoubleLegSupport', 'R-oneLegStanceStride', 'R-doubleLegSupportStride', 'R-weg'
                          , 'R-meanStrideLen', 'R-meanStepHeight', 'R-aufsetzwinkel', 'R-abrollwinkel', 'R-velocity', 'R-velocity2', 'R-cadence', 'R-OKtilt', 'R-ROMShoulderX'
                          , 'R-ROMShoulderY', 'R-ROMShoulderZ', 'R-meanROMHipX', 'R-meanROMSpineX', 'R-meanROMHipY', 'R-meanROMSpineY', 'R-meanROMHipZ', 'R-meanROMSpineZ', 'R-StrideLength'
                          , 'R-StrideLength2', 'R-StepLength', 'S-lastStep', 'S-meanAmplMidSwing', 'S-amplDiffTCIC', 'S-meanStride', 'S-meanStance', 'S-meanSwing', 'S-SwingStride', 'S-StanceStride'
                          , 'S-meanOneLegStance', 'S-meanDoubleLegSupport', 'S-oneLegStanceStride', 'S-doubleLegSupportStride', 'S-weg', 'S-meanStrideLen', 'S-meanStepHeight', 'S-aufsetzwinkel'
                          , 'S-abrollwinkel', 'S-cadence', 'S-meanROMHipX', 'S-meanROMSpineX', 'S-meanROMHipY', 'S-meanROMSpineY', 'S-meanROMHipZ', 'S-meanROMSpineZ', 'S-StrideLength2', 'S-StepLength'
                          , 'P-Altersklasse', 'P-Geschlecht', 'Pid'])
    
    dfreadmeasure = readmeasure.rename(columns={'Pid': 'P-KennungAnonym'})
    dfreadmeasure = readmeasure.rename(columns={'Pid': 'P-KennungAnonym'})
    return testMeasure, trainMeasure


create_test_train_data()