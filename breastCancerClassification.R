# Check all necessary libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
library(funModeling)


#################################################
#  Breast Cancer Project Code 
################################################

#### Data Loading ####
# Wisconsin Breast Cancer Diagnostic Dataset
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2
# Loading the csv data file from my github account

wbcd <- read.csv("https://raw.githubusercontent.com/cmrad/MLProject/master/data.csv")


# General Data Info

str(wbcd)
summary(wbcd)
## So we have 569 observations with 32 variables. 


# Check for missing values

map_int(wbcd, function(.x) sum(is.na(.x)))
## no missing values

### Data Exploration ###

# The target feature is diagnosis with levels "M" (malignant) and "B" (Benign).
# Computing the proportions
round(prop.table(table(wbcd$diagnosis)), digits = 2)
## The target variable is slightly unbalanced

# Distribution of the  Diagnosis COlumn
options(repr.plot.width=4, repr.plot.height=4)
ggplot(wbcd, aes(x=diagnosis))+geom_bar(fill="black",alpha=0.5)+theme_bw()+labs(title="Distribution of Diagnosis")

# Plotting Numerical Data

plot_num(wbcd %>%select(-id), bins=10) 


# Check for the variables' correlations. 
# Most ML algorithms assume that the predictor variables are independent from each others
# Next step: Remove mutlicollinearity (i.e. remove highly correlated predictors) for the anlysis to be robust 
wbcd_corr <- cor(wbcd %>% select(-id, -diagnosis))
corrplot::corrplot(wbcd_corr, order = "hclust", tl.cex = 0.8, addrect = 8)

## Data Transformation ## 

# The findcorrelation() function from caret package removes highly correlated predictors
# based on whose correlation is above 0.9. This function uses a heuristic algorithm 
# to determine which variable should be removed instead of selecting blindly
wbcd2 <- wbcd %>% select(-findCorrelation(wbcd_corr, cutoff = 0.9))

#Number of columns for our new data frame
ncol(wbcd2)
## transformed dataset wbcd2 is 10 variables shorter

## Data Pre-Processing ##

# Principle component analysis 
## Remove the id & diagnosis variable, then scale & center the variables

preproc_pca_wbcd <- prcomp(wbcd %>% select(-id, -diagnosis), scale = TRUE, center = TRUE)
summary(preproc_pca_wbcd)

# Compute the proportion of variance explained
pca_wbcd_var <- preproc_pca_wbcd$sdev^2
pve_wbcd <- pca_wbcd_var / sum(pca_wbcd_var)
cum_pve <- cumsum(pve_wbcd)# Cummulative percent explained
pve_table <- tibble(comp = seq(1:ncol(wbcd %>% select(-id, -diagnosis))), pve_wbcd, cum_pve)

#95% of the variance is explained with 10 PC's in the original dataset
ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0)

## Redo the above on the transformed dataset wbcd2

preproc_pca_wbcd2 <- prcomp(wbcd2, scale = TRUE, center = TRUE)
summary(preproc_pca_wbcd2)

pca_wbcd2_var <- preproc_pca_wbcd2$sdev^2

# proportion of variance explained
pve_wbcd2 <- pca_wbcd2_var / sum(pca_wbcd2_var)
cum_pve_wbcd2 <- cumsum(pve_wbcd2)
pve_table_wbcd2 <- tibble(comp = seq(1:ncol(wbcd2)), pve_wbcd2, cum_pve_wbcd2)

#  95% of the variance is explained with 8 PC's in the transformed dataset

ggplot(pve_table_wbcd2, aes(x = comp, y = cum_pve_wbcd2)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0)

# Visualize the most influential variables on the first 2 components

autoplot(preproc_pca_wbcd2, data = wbcd,  colour = 'diagnosis',
         loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")

# Visualize the first 3 components

wbcd_pcs <- cbind(as_tibble(wbcd$diagnosis), as_tibble(preproc_pca_wbcd2$x))
GGally::ggpairs(wbcd_pcs, columns = 2:4, ggplot2::aes(color = value))
## first 3 principal components separate the two classes to some extent only 
##this is expected since the variance explained by these components is not large

# LDA version
## LDA takes in consideration the different classes & could get better results

preproc_lda_wbcd <- MASS::lda(diagnosis ~., data = wbcd, center = TRUE, scale = TRUE)
preproc_lda_wbcd

## Dataframe of the LDA for visualization purposes

predict_lda_wbcd <- predict(preproc_lda_wbcd, wbcd)$x %>% 
  as_data_frame() %>% 
  cbind(diagnosis = wbcd$diagnosis)

### Data Modelling ###

# Split the dataset into train (80%) & test(20%) sets

set.seed(1815)
wbcd3 <- cbind(diagnosis = wbcd$diagnosis, wbcd2)
wbcd_sampling_index <- createDataPartition(wbcd3$diagnosis, times = 1, p = 0.8, list = FALSE)
wbcd_training <- wbcd3[wbcd_sampling_index, ]
wbcd_testing <-  wbcd3[-wbcd_sampling_index, ]

# trainControl function is used to Control the computational nuances of the train function
wbcd_control <- trainControl(method="cv", #the resampling method k-fold cross validation
                           number = 15,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# Naive Bayes Model

model_nb_wbcd <- train(diagnosis~.,
                  wbcd_training,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),# to normalize the data
                  trace=FALSE,
                  trControl=wbcd_control)

prediction_nb_wbcd<-predict(model_nb_wbcd, wbcd_testing)

## check results
cm_nb_wbcd<- confusionMatrix(prediction_nb_wbcd, wbcd_testing$diagnosis, positive = "M")
cm_nb_wbcd

## A needle plot of the naive bayes variable importance values
plot(varImp(model_nb_wbcd), top = 10, main = "Naive Bayes")


# Logistic Regression Model

model_logreg_wbcd <- train(diagnosis ~., data = wbcd_training, method = "glm", 
                         metric = "ROC", 
                         preProcess = c("scale", "center"), #to normalize the data
                         trControl = wbcd_control)

prediction_logreg_wbcd <- predict(model_logreg_wbcd, wbcd_testing)

## check results
cm_logreg_wbcd <- confusionMatrix(prediction_logreg_wbcd, wbcd_testing$diagnosis, positive = "M")
cm_logreg_wbcd

## glmnet is used as it incorporates various linear algorithms
## The below code could take some time

model_glmnet_wbcd <- train(diagnosis ~., data = wbcd_training, method = "glmnet", 
                         metric = "ROC", preProcess = c("scale", "center"), tuneLength = 20, 
                         trControl = wbcd_control)


prediction_glmnet_wbcd <- predict(model_glmnet_wbcd, wbcd_testing)

## check results
cm_glmnet_wbcd <- confusionMatrix(prediction_glmnet_wbcd, wbcd_testing$diagnosis, positive = "M")
cm_glmnet_wbcd

## A needle plot of the glmnet variable importance values

plot(varImp(model_glmnet_wbcd), top = 10, main = "glmnet")

# Random Forest Model

model_rf_wbcd <- train(diagnosis ~., data = wbcd_training,
                     method = "rf", 
                     metric = 'ROC', 
                     trControl = wbcd_control)

prediction_rf_wbcd <- predict(model_rf_wbcd, wbcd_testing)

## check results
cm_rf_wbcd <- confusionMatrix(prediction_rf_wbcd, wbcd_testing$diagnosis, positive = "M")
cm_rf_wbcd

## A needle plot of the Random Forest variable importance values

plot(varImp(model_rf_wbcd), top = 10, main = "Random forest")

# KNN Model

model_knn_wbcd <- train(diagnosis ~., data = wbcd_training, 
                      method = "knn", 
                      metric = "ROC", 
                      preProcess = c("scale", "center"), #to normalize the data 
                      trControl = wbcd_control, 
                      tuneLength =31)#to specify the number of possible k values to evaluate
## KNN model plot
plot(model_knn_wbcd)
#ROC was used to select the optimal model using the largest value.


prediction_knn_wbcd <- predict(model_knn_wbcd, wbcd_testing)

## check results
cm_knn_wbcd <- confusionMatrix(prediction_knn_wbcd, wbcd_testing$diagnosis, positive = "M")

cm_knn_wbcd

## A needle plot of the KNN variable importance values

plot(varImp(model_knn_wbcd), top = 10, main = "KNN")

# Neural Network with PCA Model
## The below code could take some time 

model_nnetpca_wbcd <- train(diagnosis ~., wbcd_training, 
                            method = "nnet", 
                            metric = "ROC", 
                            preProcess=c('center', 'scale', 'pca'), #to normalize the data
                            tuneLength = 10, 
                            trace = FALSE, 
                            trControl = wbcd_control)

prediction_nnetpca_wbcd <- predict(model_nnetpca_wbcd, wbcd_testing)

## check results
cm_nnetpca_wbcd <- confusionMatrix(prediction_nnetpca_wbcd, wbcd_testing$diagnosis, positive = "M")
cm_nnetpca_wbcd

## A needle plot of the Neural Network with PCA variable importance values

plot(varImp(model_nnetpca_wbcd), top = 8, main = "Neural Network with PCA")


# Neural Network with LDA Model

lda_training <- predict_lda_wbcd[wbcd_sampling_index, ]
lda_testing <- predict_lda_wbcd[-wbcd_sampling_index, ]
## The below code could take some time
model_nnetlda_wbcd <- train(diagnosis ~., lda_training, 
                          method = "nnet", 
                          metric = "ROC", 
                          preProcess = c("center", "scale"), #to normalize the data
                          tuneLength = 10, 
                          trace = FALSE, 
                          trControl = wbcd_control)

prediction_nnetlda_wbcd <- predict(model_nnetlda_wbcd, lda_testing)

## check results
cm_nnetlda_wbcd <- confusionMatrix(prediction_nnetlda_wbcd, lda_testing$diagnosis, positive = "M")
cm_nnetlda_wbcd

#### Results ####
#### Model Evaluation ####

model_list <- list(Naive_Bayes=model_nb_wbcd,logisic = model_logreg_wbcd, glmnet = model_glmnet_wbcd,
                   rf = model_rf_wbcd,KNN=model_knn_wbcd,
                   Neural_with_LDA = model_nnetlda_wbcd,Neural_with_PCA = model_nnetpca_wbcd)
models_results <- resamples(model_list)

summary(models_results)

## Some models have high variability depending on the processed sample (Naive_Bayes & logistic reg)
## The model Neural with LDA achieve a great auc with some variability.
##The ROC metric measure the auc of the roc curve of each model; this metric is independent of any threshold.

bwplot(models_results, metric = "ROC")

# Models result with the testing dataset. 
#Prediction classes are obtained by default with a threshold of 0.5 which isn't ideal with an unbalanced dataset like this.


cm_list <- list(cm_naive_bayes=cm_nb_wbcd,cm_rf = cm_rf_wbcd, cm_logisic = cm_logreg_wbcd,
                cm_KNN=cm_knn_wbcd,cm_nnet_LDA = cm_nnetlda_wbcd,cm_nnet_pca = cm_nnetpca_wbcd)

results <- sapply(cm_list, function(x) x$byClass) 


results%>% knitr::kable()

# The neural network model with LDA yields the optimal results for sensitivity (detection of breast cancer cases) 
# along with a balanced accuracy and F1 score of 0.988 & 0.987, respectively

cm_results_max <- apply(results, 1, which.is.max)

output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(results)[cm_results_max],
                            value=mapply(function(x,y) {results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report


## Print the direct accuracy of the chosen model (NNet with LDA)

paste0(round(mean(prediction_nnetlda_wbcd == wbcd_testing$diagnosis)*100, digits=4),"%")


## View the final predictions 

View(prediction_nnetlda_wbcd)

