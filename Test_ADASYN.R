## Projet apprentissage d?s?quilibr?
## Vishwa Elankumaran et No? Lebreton
## 10/2021

##################################################################################
## Mise en place des mod?les et des m?triques obtenues avec la pr?diction
##################################################################################

library(MASS)
library(caret)
library(rpart)
library(glmnet)
library(doParallel)


setwd("/Users/vishwa/Documents/Master\ Informatique\ -\ Data\ Mining/Advanced\ Supervised\ Learning/Projet/Apprentissage\ Deséquilibré/")


## Chargement des donn?es ?quilibr?es et d?s?quilibr?es ####
data_unbalanced = read.csv("data_unbalanced.csv",sep = ";",header = T)

data_balanced = list()
for (i in 1:10){
  fichier = paste("data_adasyn_0",i,".csv",sep="")
  data_balanced[[i]] = read.table(fichier,sep=";")
  colnames(data_balanced[[i]])[ncol(data_balanced[[i]])] = "default_ind"
  
}

## Ratio entre les classes des donn?es d?s?quilibr?es et ?quilibr?es ####
for(i in 1:10){
  if(i == 1){
    d0_unbalanced = length(which(data_unbalanced$default_ind==0))/nrow(data_unbalanced)
    d1_unbalanced = length(which(data_unbalanced$default_ind==1))/nrow(data_unbalanced)
    print(paste("Pour les donn?es d?s?quilibr?es, minority = ",round(d1_unbalanced,3)," et majority = ",round(d0_unbalanced,3),"(",round(d0_unbalanced/d1_unbalanced,2),":1)"))
  }
  d0_balanced = length(which(data_balanced[[i]]$default_ind==0))/nrow(data_balanced[[i]])
  d1_balanced = length(which(data_balanced[[i]]$default_ind==1))/nrow(data_balanced[[i]])
  print(paste("Pour beta = ",i/10,", minority = ",round(d1_balanced,3)," et majority = ",round(d0_balanced,3),"(",round(d0_balanced/d1_balanced,2),":1)"))
}

## Fonction permettant d'entrainer et de pr?dire les valeurs de Y ####

f_test = function(data,pourcentage_train){
  data$default_ind = as.factor(data$default_ind)
  train_index = sample(1:nrow(data),nrow(data)*pourcentage_train)
  data_train = data[train_index,]
  data_test = data[-train_index,]
  
  ## ADL ##
  fit_lda = lda(default_ind~.,data = data_train)
  pred_lda = predict(fit_lda,newdata = data_test)
  
  ## Arbre de d?cision ##
  arbre = rpart(default_ind~.,data = data_train)
  pred_arbre = predict(arbre,newdata = data_test,type="class")
  
  return(list(data_test$default_ind,pred_lda$class,pred_arbre))
}

## Application des diff?rents mod?les ####
pourcentage_train = 0.8

registerDoParallel(cores = detectCores()-1)
t0_dopar = Sys.time()
res_balanced = foreach(i = 1:10,.packages = c("MASS","rpart")) %dopar% f_test(data_balanced[[i]],pourcentage_train = pourcentage_train)
t1_dopar = Sys.time()
t1_dopar-t0_dopar # Time difference of 6.654092 mins
stopImplicitCluster()

res_unbalanced = f_test(data_unbalanced,pourcentage_train = pourcentage_train)

res_balanced[[11]] = res_unbalanced

## Fonction permettant de calculer les m?triques ####

f_metrics = function(res){
  mc = list()
  precision = list()
  recall= list()
  F1_score = list()
  Kappa = list()
  for(i in 1:2){
    mc[[i]] = table(res[[i+1]],res[[1]])
    precision[[i]] = mc[[i]][2,2]/(mc[[i]][2,2]+mc[[i]][2,1])
    recall[[i]] = mc[[i]][2,2]/(mc[[i]][2,2]+mc[[i]][1,2])
    F1_score[[i]] = 2*((precision[[i]]*recall[[i]])/(precision[[i]]+recall[[i]]))
    p_0 = ((mc[[i]][1,1]+mc[[i]][1,2])/sum(mc[[i]]))*((mc[[i]][1,1]+mc[[i]][2,1])/sum(mc[[i]]))
    p_1 = ((mc[[i]][2,1]+mc[[i]][2,2])/sum(mc[[i]]))*((mc[[i]][1,2]+mc[[i]][2,2])/sum(mc[[i]]))
    p_o = ((mc[[i]][1,1]+mc[[i]][2,2])/sum(mc[[i]]))
    Kappa[[i]] = (p_o-(p_0+p_1))/(1-(p_0+p_1))
  }
  return(list(mc,precision,recall,F1_score,Kappa))
}

## Calcul des m?triques ####

registerDoParallel(cores = detectCores()-1)
t0_dopar = Sys.time()
res_metrics = foreach(j = 1:11) %dopar% f_metrics(res_balanced[[j]])
t1_dopar = Sys.time()
t1_dopar-t0_dopar # Time difference of 10.15873 secs
stopImplicitCluster()

## Mise en forme des metrics ####

metrics_to_df = function(model){
  res_matrix = matrix(0,ncol = 4,nrow = 12)
  for(i in 2:12){
    temp = c()
    for(j in 2:5){
      temp[j-1] = res_metrics[[i-1]][[j]][model]
    }
    res_matrix[i,] = as.numeric(temp)
  }
  res = as.data.frame(res_matrix)
  res[1,] = res[12,]
  res = res[-12,]
  res[,5]=seq(0,1,0.1)
  colnames(res) = c("precision","recall","F1_score","Kappa","Beta")
  return(res)
}

res_adl = metrics_to_df(1)
res_arbre = metrics_to_df(2)

## Repr?sentation graphique des m?triques ####


library(ggplot2)
library(tidyr)

res_adl_trans = gather(data = res_adl,key = Metrics,value = Values,precision,recall,F1_score,Kappa)
res_arbre_trans = gather(data = res_arbre,key = Metrics,value = Values,precision,recall,F1_score,Kappa)

ggplot(res_adl_trans) + ggtitle("Evolution des m?triques en fonction de beta pour l'ADL") + aes(x = Beta, y = Values, color = Metrics) + geom_line()
ggplot(res_arbre_trans) + ggtitle("Evolution des m?triques en fonction de beta pour l'arbre de d?cision") + aes(x = Beta, y = Values, color = Metrics) + geom_line()

## Pourcentage dans les matrices de confusion ####

library(gridExtra)

pourcentage_mc = function(res_metrics,methode){
  #methode = 1 si ADL et 2 si arbre
  pourcentage_1_1 = pourcentage_1_2 = pourcentage_2_1 =pourcentage_2_2 = c()
  if(methode == 1){nom_m = "ADL"}else{nom_m="arbre de d?cision"}
  
  for(i in 1:10){
    if(i==1){
      mc_pourentage = prop.table(as.table(res_metrics[[11]][[1]][methode][[1]]))
      pourcentage_1_1 = c(pourcentage_1_1,mc_pourentage[1,1])
      pourcentage_1_2 = c(pourcentage_1_2,mc_pourentage[1,2])
      pourcentage_2_1 = c(pourcentage_2_1,mc_pourentage[2,1])
      pourcentage_2_2 = c(pourcentage_2_2,mc_pourentage[2,2])
    }
    mc_pourentage = prop.table(as.table(res_metrics[[i]][[1]][methode][[1]]))
    pourcentage_1_1 = c(pourcentage_1_1,mc_pourentage[1,1])
    pourcentage_1_2 = c(pourcentage_1_2,mc_pourentage[1,2])
    pourcentage_2_1 = c(pourcentage_2_1,mc_pourentage[2,1])
    pourcentage_2_2 = c(pourcentage_2_2,mc_pourentage[2,2])
  }
  
  nom_col = c("Pourcentage","Beta")
  beta = seq(0,1,0.1)
  res_pourcentage_1_1 = data.frame(pourcentage_1_1,beta)
  res_pourcentage_1_2 = data.frame(pourcentage_1_2,beta)
  res_pourcentage_2_1 = data.frame(pourcentage_2_1,beta)
  res_pourcentage_2_2 = data.frame(pourcentage_2_2,beta)
  colnames(res_pourcentage_1_1) = colnames(res_pourcentage_1_2) = colnames(res_pourcentage_2_1) = colnames(res_pourcentage_2_2) = nom_col
  
  g1 = ggplot(res_pourcentage_1_1) + ggtitle(paste("Evolution des pourcentages mc[1,1] en fonction de beta pour l'",nom_m)) + aes(x = Beta, y = Pourcentage) + geom_line()
  g2 = ggplot(res_pourcentage_1_2) + ggtitle(paste("Evolution des pourcentages mc[1,2] en fonction de beta pour l'",nom_m)) + aes(x = Beta, y = Pourcentage) + geom_line()
  g3 = ggplot(res_pourcentage_2_1) + ggtitle(paste("Evolution des pourcentages mc[2,1] en fonction de beta pour l'",nom_m)) + aes(x = Beta, y = Pourcentage) + geom_line()
  g4 = ggplot(res_pourcentage_2_2) + ggtitle(paste("Evolution des pourcentages mc[2,2] en fonction de beta pour l'",nom_m)) + aes(x = Beta, y = Pourcentage) + geom_line()
  
  grid.arrange(g1,g2,g3,g4, ncol=2, nrow = 2)
  
}

pourcentage_mc(res_metrics,1)
pourcentage_mc(res_metrics,2)

## Graphique matrice de confusion ####

con_lda = confusionMatrix(res_balanced[[10]][[2]], res_balanced[[10]][[1]])
fourfoldplot(con_lda$table, color = c("cyan", "pink"),conf.level = 0, margin = 1, main = "Confusion Matrix")

con_arbre = confusionMatrix(res_balanced[[10]][[3]], res_balanced[[10]][[1]])
fourfoldplot(con_arbre$table, color = c("cyan", "pink"),conf.level = 0, margin = 1, main = "Confusion Matrix")
