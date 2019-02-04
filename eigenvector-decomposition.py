############################################
################ LIBRARY IMPORTS  
############################################
import pandas as pd
import numpy as np
   
############################################
################ LOAD DATA 
############################################
crime_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data',sep=',',names=['state','county', 'community','communityname','fold','population','householdsize','racepctblack','racePctWhite','racePctAsian','racePctHisp','agePct12t21','agePct12t29','agePct16t24','agePct65up','numbUrban','pctUrban','medIncome','pctWWage','pctWFarmSelf','pctWInvInc','pctWSocSec','pctWPubAsst','pctWRetire','medFamInc','perCapInc','whitePerCap','blackPerCap','indianPerCap','AsianPerCap','OtherPerCap','HispPerCap','NumUnderPov','PctPopUnderPov','PctLess9thGrade','PctNotHSGrad','PctBSorMore','PctUnemployed','PctEmploy','PctEmplManu','PctEmplProfServ','PctOccupManu','PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr','FemalePctDiv','TotalPctDiv','PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par','PctTeen2Par','PctWorkMomYoungKids','PctWorkMom','NumIlleg','PctIlleg','NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10','PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10','PctSpeakEnglOnly','PctNotSpeakEnglWell','PctLargHouseFam','PctLargHouseOccup','PersPerOccupHous','PerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup','PctPersDenseHous','PctHousLess3BR','MedNumBR','HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt','PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal','OwnOccHiQuart','RentLowQ','RentMedian','RentHighQ','MedRent','MedRentPctHousInc','MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','NumStreet','PctForeignBorn','PctBornSameState','PctSameHouse85','PctSameCity85','PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop','PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop','ViolentCrimesPerPop'], encoding='latin-1',engine='python',na_values=['?'])

############################################
################ PRE-PROCESSING 
############################################
# extraction of feature matrix
# and missing value replaced with mean
df=crime_data.drop(['state','county', 'community','communityname','fold','ViolentCrimesPerPop'], axis=1)
df=df.fillna(df.mean())
#print(df)

# For finding eigen values and eigen vectors the matrix should be a square matrix
# i.e. number of rows = number of columns
# here we know from the data description that
# number of rows = 1994
# number of columns = 128
# dropped column 6
# therefore, we need to drop all the rows except 128 rows
# I am choosing to keep top 122 

df=df.values
m = np.asmatrix(df)
m = m[:122]
print(m)

############################################
################ eigen values & eigen vectors
############################################
eigenvalues = np.linalg.eigvals(m)
print("eigenvalues = ", eigenvalues)

values, vectors = np.linalg.eig(m)

print("val = ", values)
print("vec = ", vectors)

