## Projet apprentissage d�s�quilibr�
## Vishwa Elankumaran et No� Lebreton
## 10/2021

##################################################################################
## Simulation ADASYN pour des beta diff�rents
##################################################################################


setwd("C:/Users/nlebr/Desktop/projet_apprentissage_desequilibre")
source("ADASYN.r")

data = read.csv("data.csv",sep = ",",header = T)

#s�lection des variables importantes (https://www.kaggle.com/rameshmehta/credit-risk-analysis)

data = data[,c("loan_amnt","int_rate","grade","annual_inc","purpose","installment","term","default_ind")]

data_explicative = data[,1:(ncol(data)-1)]

library("FactoMineR")
library("factoextra")

## Analyse Factorielle pour g�rer les variables qualitatives ####

res_famd = FAMD(data_explicative, graph = FALSE,ncp = 7)
eig_val = get_eigenvalue(res_famd)

data_famd = as.data.frame(res_famd$ind$coord)
data_unbalanced = cbind(data_famd,data$default_ind)

colnames(data_unbalanced)[8] = "default_ind"

write.table(data_unbalanced,"data_unbalanced.csv",sep=";")

## Simulation pour des beta diff�rents ####

b = seq(0.1,1,0.1)
for (i in 1:length(b)){
  t0 = Sys.time()
  data_balanced = ADASYN(data_unbalanced,d_th = 1,beta = b[i],K = 5)
  t1 = Sys.time()
  print(i)
  print(t1-t0)
  fichier = paste("data_adasyn_0",i,".csv",sep="")
  write.table(data_balanced,fichier,sep=";")
}




