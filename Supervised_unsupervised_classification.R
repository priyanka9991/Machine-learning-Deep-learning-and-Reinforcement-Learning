###### PRIYANKA VASANTHAKUMARI #####
## Supervised and Unsupervised Classification ##

###### Part 1 - Supervised learning

library(magrittr)
library(tidyverse)
library(caret)
library(MASS)
library(boot)
library(klaR)

load("/Users/priyanka/Documents/Course works/Data Mining /Final Project/class_data.RData")
data <- data.frame(x,y) 

##### No feature selection

#Logistic regression 
glm.fit=glm(y~.,data=data,family=binomial,maxit=100)
summary(glm.fit)
#Cross validation with LR
set.seed(5)
glmcv<-cv.glm(data = data, glm.fit,K=10)
glmcv$delta   #CV error
##A vector of length two. The first component is the raw cross-validation estimate of prediction error. 
#The second component is the adjusted cross-validation estimate. 
#The adjustment is designed to compensate for the bias introduced by not using leave-one-out cross-validation.
1-glmcv$delta  # Accuracy
train_control <- trainControl(method="cv", number=10)
set.seed(5)
model <- caret::train(as.factor(y)~., data=data, trControl=train_control, method="glm")
print(model)


#LDA 
require(MASS)
train_control <- trainControl(method="cv", number=10)
set.seed(5)
model <- caret::train(as.factor(y)~., data=data, trControl=train_control, method="lda")
print(model)

#SVM 
library(e1071)
data$y <- as.factor(data$y)
svmfit=svm(data$y~.,data=data, kernel="linear", cost=1)
set.seed(5) #Linear
model <- caret::train(as.factor(y)~., data=data, trControl=train_control, method="svmLinear")
print(model) 

set.seed(5) # Radial
model <- caret::train(as.factor(y)~., data=data, trControl=train_control, method="svmRadial")
print(model) 

## 3 KNN & Cross-validation

source("/Users/priyanka/Documents/Course works/Data Mining /my.cv.knn.R")
knn_x <- x  # Feature set
knn_y <- data$y   # Labels
k1 <- c( 2, 5, 10, 20, 50, 100, 150, 200, 300) # KNN tuning parameter k
nk=length(k1)
class_error=rep(0,nk)  #Misclassification error 

# Crossvalidation across all values of k (tuning parameters)
for (i in 1:nk){
  k2=as.integer(k1[i])
  class_error[i]<- my.cv.knn(k2,knn_x,knn_y,10)   # 10-fold cross-validation
}
# Scatter plot
plot(k1,class_error,xlab="k", ylab="Misclassification error")  
# Line plot
lines(k1,class_error,xlab="k", ylab="Misclassification error") 

## 4 Tuning k - Choosing k correponding to minimum Misclassification error
k_opt = k1[which.min(class_error)]
# Optimum value of k
k_opt
1-min(class_error) # CV Accuracy

#### Random forests on whole data and feature selection with importance

library(tree)
library(randomForest)
library(gbm)
set.seed(5)
data.rf <- randomForest(as.factor(y) ~ ., data=data, ntree=1000,
                        keep.forest=FALSE, importance=TRUE)

varImpPlot(data.rf,main="Importance of variables") # Importance plot
imp <- data.rf$importance
imp_sort <- imp[order(-imp[,2]),]    # Sort variables in descending order of importance
flag = rep(0, 500)

# RF error on all the features

train=sample(1:nrow(data),200)
oob.err=double(500)
test.err=double(500)
for(mtry in 1:500){
  fit=randomForest(y~.,data=data,subset=train,mtry=mtry,ntree=400)
  oob.err[mtry]=fit$mse[400]#Mean squared error for 400 trees
  pred=predict(fit,data[-train,])
  test.err[mtry]=mean((data[-train,]$y-pred)^2)
}
#  81.44% accuracy in random forests 

matplot(1:500,cbind(test.err,oob.err),pch=19,col=c("red","blue"),type="b",ylab="Mean Squared Error")
legend("topright",legend=c("Test", "OOB"),pch=19,col=c("red","blue"))
min(oob.err)
min(test.err)

### Feature Selection - Using RF variable importance & select the percentage of important variables

for(i in 1:500){
  flag[i] <- sum(imp_sort[1:i,2])<0.9*sum(imp_sort[,2]) # 0.7 corresponds to 70% of variables
}
imp_var <- rownames(imp_sort[1:sum(flag),])
sel_feature<-data[,imp_var]
sel_feature <- sapply(sel_feature, function(p) as.numeric(unlist(p)))
newdatarf <- data.frame(sel_feature,y)# data frame of selected features from Variable importance

#Logistic regression after RF imp selection

glm.fit=glm(y~.,data=newdatarf,family=binomial,maxit=100)
summary(glm.fit)
#10 fold Cross validation with LR
set.seed(5)
glmcv<-cv.glm(data = newdatarf, glm.fit,K=10)
glmcv$delta   #CV error
##A vector of length two. The first component is the raw cross-validation estimate of prediction error. 
#The second component is the adjusted cross-validation estimate. 
#The adjustment is designed to compensate for the bias introduced by not using leave-one-out cross-validation.
1-glmcv$delta  # CV Accuracy


#LDA after RF imp selection
require(MASS)
lda.fit=lda(y~.,data=newdatarf)
plot(lda.fit)
# Crossvalidation
set.seed(5)
train_control <- trainControl(method="cv", number=10)
model <- caret::train(as.factor(y)~., data=newdatarf, trControl=train_control, method="lda")
# summarize results
print(model) #CV Accuracy

##SVM after RF imp selection
library(e1071)
svmfit=svm(as.factor(y)~.,data=newdatarf, kernel="linear", cost=1)
set.seed(5) # Linear
model <- caret::train(as.factor(y)~., data=newdatarf, trControl=train_control, method="svmLinear")
print(model)
set.seed(5) # Radial
train_control <- trainControl(method="cv", number=10)
model <- caret::train(as.factor(y)~., data=newdatarf, trControl=train_control, method="svmRadial")
# summarize results
print(model)

#QDA after RF var sel
require(MASS)
qda.fit=qda(as.factor(y)~.,data=newdatarf)
set.seed(5)
train_control <- trainControl(method="cv", number=10)
model <- caret::train(as.factor(y)~., data=newdatarf, trControl=train_control, method="qda")
print(model)


###### LASSO

library(glmnet)
p=model.matrix((y~.),data)[,-1] # take out the first column which are all 1's for intercept
y=data$y
dim(p)

set.seed(5)
glmmod <- glmnet(p, y=as.factor(y), alpha=1, family="binomial")
summary(glmmod)
# Plot variable coefficients vs. shrinkage parameter lambda.
plot(glmmod, xvar="lambda",label=TRUE)
cv.lasso=cv.glmnet(p,y,alpha = 1, family = "binomial")
summary(cv.lasso)
plot(cv.lasso)
lasso.best.lambda=cv.lasso$lambda.min # best lambda value corresponding to min cv.error
cv.lasso$lambda.1se # lambda corresponding to 2nd dashed line - 1 standard error
lasso.best.lambda
# It is to be noted that the coefficients of some of the predictors are zero
predict(glmmod, s=lasso.best.lambda, type="coefficients")
model <- glmnet(p, y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
summary(model)

# Select non zero coefficients after Lasso
tmp_coef <- nonzeroCoef(model$beta, bystep = FALSE)
selected_var <- p[,tmp_coef]       # Contains only the non-zero coefficients
newdata <- data.frame(selected_var,y) # New dataframe containng the selected variables after LASSO

#Logistic regression after LASSO
glm.fit=glm(as.factor(y)~.,data=newdata,family=binomial)
summary(glm.fit)
#Cross validation with LR
set.seed(5)
glmcv<-cv.glm(data = newdata, glm.fit,K=5)
glmcv$delta   #CV error
1-glmcv$delta  # CV Accuracy

#LDA after LASSO
require(MASS)
lda.fit=lda(as.factor(y)~.,data=newdata)
plot(lda.fit)
set.seed(5)
train_control <- trainControl(method="cv", number=10)
model <- caret::train(as.factor(y)~., data=newdata, trControl=train_control, method="lda")
print(model)


#QDA after LASSO
require(MASS)
qda.fit=qda(as.factor(y)~.,data=newdata)
set.seed(5)
train_control <- trainControl(method="cv", number=10)
model <- caret::train(as.factor(y)~., data=newdata, trControl=train_control, method="qda")
print(model)


#SVM after LASSO
library(e1071)
svmfit=svm(newdata$y~.,data=newdata, kernel="linear", cost=1)
set.seed(5)
model <- caret::train(as.factor(y)~., data=newdata, trControl=train_control, method="svmLinear")
print(model)#CV Accuracy linear
set.seed(5)
model <- caret::train(as.factor(y)~., data=newdata, trControl=train_control, method="svmRadial")
print(model) #CV Accuracy radial

## Feature selection 3

library (FSelector)
trainTask <- makeClassifTask(data = data,target = "y",positive = "1")
trainTask
trainTask <- normalizeFeatures(trainTask,method = "standardize")

#Sequential Forward Search - SVM Radial
library (mlr)
library(dplyr)
ctrl = makeFeatSelControlSequential(method = "sfs", alpha = 0.02)
rdesc = makeResampleDesc("CV", iters = 10)
sfeats = selectFeatures(learner = "classif.svm", task = trainTask, resampling = rdesc, control = ctrl,
                        show.info = FALSE)    # default is svm radial

sel_var_sfs <- data %>% select(one_of(sfeats$x))
set.seed(5)
model <- caret::train(data.frame(sel_var_sfs), as.factor(y), trControl=train_control, method="svmRadial")
print(model) #86.52 %

#Sequential Forward Method-knn
ctrl = makeFeatSelControlSequential(method = "sfs", alpha = 0.02)
rdesc = makeResampleDesc("CV", iters = 10)
sfeats_knn = selectFeatures(learner = "classif.knn", task = trainTask, resampling = rdesc, control = ctrl,
                            show.info = FALSE)  
sel_var_sfs_knn <- data %>% select(one_of(sfeats_knn$x))
set.seed(5)
model <- caret::train(data.frame(sel_var_sfs_knn), as.factor(y), trControl=train_control, method="knn")
print(model)

#Sequential Forward Floating Search - SVM Radial
ctrl = makeFeatSelControlSequential(method = "sffs", alpha = 0.02)
rdesc = makeResampleDesc("CV", iters = 10)
sfeats_sffs = selectFeatures(learner = "classif.svm", task = trainTask, resampling = rdesc, control = ctrl,
                             show.info = FALSE)

sel_var_sff <- data %>% select(one_of(sfeats_sffs$x))
set.seed(5)
model <- caret::train(data.frame(sel_var_sff), as.factor(y), trControl=train_control, method="svmRadial")
print(model) # 86.77 

#Sequential Floating Forward Search - LDA/QDA - Only 66.78 %

#Sequential Floating Forward Method - KNN
ctrl = makeFeatSelControlSequential(method = "sffs", alpha = 0.02)
rdesc = makeResampleDesc("CV", iters = 10)
sfeats_sffs_knn = selectFeatures(learner = "classif.knn", task = trainTask, resampling = rdesc, control = ctrl,
                                 show.info = FALSE)
sel_var_sff_knn <- data %>% select(one_of(sfeats_sffs_knn$x))
set.seed(5)
model <- caret::train(data.frame(sel_var_sff_knn), as.factor(y), trControl=train_control, method="knn")
print(model)
test_err <- 1 - max(model$results$Accuracy)  # Testing error estimate

# Generating y_new
data_final <- data.frame(sel_var_sff,y)
svm_final = svm(as.factor(y)~.,data=data_final, kernel="radial", cost=1)
ynew=predict(svm_final, xnew)
ynew
save(ynew,test_err,file="Sup_results.RData")




#################################################################
##################################################################
##### PART 2- UNSUPERVISED LEARNING ##########

load("/Users/priyanka/Documents/Course works/Data Mining /Final Project/cluster_data.RData")
dim(y)

# Heirarchial Clustering - to visualise the dendrogram
library(mclust)
hc.complete=hclust(dist(y),method="complete")    
plot(hc.complete)

## FEATURE SELECTION ##

#tSNE feature selection
library(Rtsne)
set.seed(1)
tsne <- Rtsne(scale(y), dims = 2, perplexity=30, verbose=TRUE, max_iter = 1000)
plot(tsne$Y[,1],tsne$Y[,2])     # Selected features

#K-means with tsne variables with 5 clusters - Better visualization
tsne_x<-as.matrix(tsne$Y)
set.seed(5)
km.out=kmeans(tsne_x,5,nstart=15)
km.out$cluster
plot(tsne_x,col=km.out$cluster,cex=2,pch=1,lwd=2,xlab='t-SNE feature 1',ylab='t-SNE feature 2', main='k means clustering on t-SNE features')

#Isomap feature selection
library(vegan)
dis <- vegdist(y) # generating dissimiliarities
set.seed(5)
simData_dim2_IM = isomap(dis, dims=10, k=3)
dim(simData_dim2_IM$points )    # Selected features

## Sammon mapping feature selection
library(Rdimtools)
set.seed(5)
sam <- do.sammon(y, ndim = 5, preprocess = c("null", "center", "scale",
                                             "cscale", "decorrelate", "whiten"), initialize = c("random", "pca"))
sam$Y  # Selected features

#PCA 
set.seed(5)
y.pca <- prcomp(y, center = TRUE,scale. = TRUE) 
summary(y.pca)
library(devtools)
library(ggbiplot)
biplot(y.pca,scale =0)
std_dev <- y.pca$sde
#compute variance
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
str(y.pca)
y.pca$x   # Principle components to be selected

## CLUSTER SELECTION METHODS ##


# Elbow method - inbuilt package
library(factoextra)   
set.seed(5)
fviz_nbclust(y.pca$x[,1:10], kmeans, method = "wss") +     # Change the feature set depending on the method PCA/tsne/sammon
  geom_vline(xintercept = 5, linetype = 2)+
  labs(subtitle = "Elbow method - PCA - 10 componets")


#Elbow Method for finding the optimal number of clusters
# Within sum of squares (WSS) is the measure
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data_elbow <- as.matrix(tsne$Y )
wss <- sapply(1:k.max, 
              function(k){kmeans(data_elbow, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


#Kmeans BIC AIC

kmeansAIC = function(fit){
  
  m = ncol(fit$centers)
  n = length(fit$cluster)
  k = nrow(fit$centers)
  D = fit$tot.withinss
  return(data.frame(AIC = D + 2*m*k,
                    BIC = D + log(n)*m*k))
}

K_val = c(2,3,4,5,6,7)
AIC <- rep(0, length(K_val))
BIC <- rep(0, length(K_val))
set.seed(1)
for (j in 1:length(K_val)){
  fit <- kmeans(x = y.pca$x[,1:50] ,centers = K_val[j])
  AIC_BIC<-kmeansAIC(fit)
  AIC[j]<-AIC_BIC$AIC
  BIC[j]<-AIC_BIC$BIC
}


plot(K_val, BIC,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="BIC",main="PCA 50 components")
abline(v=c(4,5),col=c("blue","red"))









