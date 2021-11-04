## Projet apprentissage déséquilibré
## Vishwa Elankumaran et Noé Lebreton
## 10/2021

## ADASYN ####

library(FNN)
library(parallel)

ADASYN = function(data_unbalanced,d_th,beta,K){
  ## d_th = taux de déséquilibre maximum toléré (d_th : [0;1])
  ## beta = niveau d'équilibre final du jeu de données. Si beta = 1, le jeu de données est "équilibré" (beta : [0;1])
  #on suppose que Y a comme valeur 0 ou 1. La classe majoritaire : 0 et la classe minoritaire : 1 
  #on suppose également que Y est la dernière variable
  
  colnames(data_unbalanced)[ncol(data_unbalanced)] = "Y"
  majority_ind = which(data_unbalanced[,ncol(data_unbalanced)]==0)
  minority_ind = which(data_unbalanced[,ncol(data_unbalanced)]==1)
  
  ms = length(minority_ind)
  ml = length(majority_ind)
  d = ms/ml 
  if(d<d_th){
    G = (ml - ms) * beta
    res_knn = get.knn(data_unbalanced[,-ncol(data_unbalanced)],K)
    res_knn_1 = as.data.frame(res_knn$nn.index)[minority_ind,]
    
    res_knn_1_liste = as.vector(t(res_knn_1))
    corresp_knn = data_unbalanced[res_knn_1_liste,ncol(data_unbalanced)]
    dim(corresp_knn) = c(5,nrow(res_knn_1))
    
    corresp_knn = as.data.frame(t(corresp_knn))
    
    delta_i = apply(corresp_knn,1,FUN = sum) #nombre d'individus parmi les K ppv faisant partie de la classe minoritaire
    r_i = (1-(delta_i/K)) #nombre d'individus de la classe majoritaire sur le nombre de ppv 

    r_i_chap = r_i/sum(r_i)
    
    #nombre de données synthétiques pour chaque xi
    g_i = round(r_i_chap*G)
    
    #parallélisation de la création des données synthétiques
    cl = makeCluster(detectCores()-1)
    inter = split(seq(1,length(minority_ind)),c(1:(detectCores()-1)))
    par.res = parLapply(cl,inter, fun = f_syn,gi=g_i,data=data_unbalanced[,-ncol(data_unbalanced)],knn=res_knn_1,corresp_knn = corresp_knn,K=K)
    stopCluster(cl)
    
    data_s = Reduce(rbind,par.res)
    data_s[,"Y"] = rep(1,nrow(data_s)) 
    return(rbind(data_s,data_unbalanced))
  }else{
    print("Les données sont équilibrées selon le paramètre d_th")
  }
}


f_syn = function(inter,gi,data,knn,corresp_knn,K){
  s_i = list()
  ind = 0
  for(i in inter){
    if(gi[i] > 0){
      ind_min = which(corresp_knn[i,]==1)
      #boucle allant de 1 jusqu'au nombre de données synthétiques à créer pour xi
      for(g in 1:gi[i]){
        #si il n'y pas de données minoritaires dans les ppv de xi alors on prend aléatoirement x_zi dans les ppv majoritaires
        if(length(ind_min)==0){
          x_zi = data[knn[i,sample(1:K,1)],]
          s_i[as.character(ind)] = list((data[i,] + (x_zi - data[i,])*runif(1,0,1)))
          ind = ind + 1
        }else{
          x_zi = data[knn[i,sample(ind_min,1)],]
          s_i[as.character(ind)] = list((data[i,] + (x_zi - data[i,])*runif(1,0,1)))
          ind = ind + 1
        }
      }
    }
  }
  return(Reduce(rbind,s_i))
}


