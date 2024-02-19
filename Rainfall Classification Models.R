install.packages("tree")
install.packages("e1071")
install.packages("ROCR")
install.packages("adabag")
install.packages("rpart")
install.packages("caret")
install.packages("neuralnet")
install.packages("gbm")

setwd("~/KaggleCompetitionData")
rm(list = ls())
WAUS <- read.csv("HumidPredict2023D.csv")
L <- as.data.frame(c(1:49))
set.seed(32505485) # Your Student ID is the random seed
L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS <- WAUS[(WAUS$Location %in% L),]
WAUS <- WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows


# - Exploration of Data

# rows with NA in MHT are removed
WAUS <- WAUS[!is.na(WAUS$MHT),]
# Count MHT=1
MHT1 <- sum(WAUS$MHT == 1)
# Count MHT=0
MHT0 <- sum(WAUS$MHT == 0)
# Proportion by dividing
finaloutput <- MHT1 / (MHT1 + MHT0)
 

# Filter numeric columns for real-valued attributes only
numeric_columns_only <- sapply(WAUS, is.numeric)
# Mean Calc
mean_values <- sapply(WAUS[, numeric_columns_only], mean, na.rm = TRUE) # im using na.rm as True to ignore NA
# Stdev Calc
std_dev_values <- sapply(WAUS[, numeric_columns_only], sd, na.rm = TRUE)
# Combine the mean and standard deviation values into a data frame
summary_stats <- data.frame(Mean = mean_values, Std_Deviation = std_dev_values)
# Print the summary statistics
print(summary_stats)


# Pre processing of the data and Splitting the data#
WAUS <- na.omit(WAUS)

set.seed(32505485) #Student ID as random seed
train.row = sample(1:nrow(WAUS), 0.7*nrow(WAUS))
WAUS.train = WAUS[train.row,]
WAUS.test = WAUS[-train.row,]

WAUS.train$WindDir9am <- as.factor(WAUS.train$WindDir9am)
WAUS.train$WindDir3pm <- as.factor(WAUS.train$WindDir3pm)
WAUS.train$RainToday <- as.factor(WAUS.train$RainToday)
WAUS.train$Location <- as.factor(WAUS.train$Location)
WAUS.train$MHT <- as.factor(WAUS.train$MHT)
WAUS.train$WindGustDir <- as.factor(WAUS.train$WindGustDir)


WAUS.test$WindDir9am <- as.factor(WAUS.test$WindDir9am)
WAUS.test$WindDir3pm <- as.factor(WAUS.test$WindDir3pm)
WAUS.test$RainToday <- as.factor(WAUS.test$RainToday)
WAUS.test$Location <- as.factor(WAUS.test$Location)
WAUS.test$MHT <- as.factor(WAUS.test$MHT)
WAUS.test$WindGustDir <- as.factor(WAUS.test$WindGustDir)



#Decision Tree
library(tree)
library(ROCR)
library(caret)
set.seed(32505485)

# - Classification Model
tree_model <- tree(MHT ~ ., data = WAUS.train)
summary(tree_model)
plot(tree_model)
text(tree_model, pretty = 0)

# - Accuracy & Confusion Matrix
tree_pred_model <- predict(tree_model, newdata = WAUS.test, type = "class")
print(tree_pred_model)
table(tree_pred_model, WAUS.test$MHT)
accuracy <- sum(tree_pred_model == WAUS.test$MHT) / nrow(WAUS.test)
accuracy

# - AUC and ROC
tree_pred <- predict(tree_model, newdata = WAUS.test, type = "vector")
tree_confidence <- tree_pred[,2]
tree_pred_rocr <- prediction(tree_confidence, WAUS.test$MHT)
tree_auc <- performance(tree_pred_rocr, "auc")@y.values[[1]]
tree_auc
plot(performance(tree_pred_rocr, "tpr", "fpr"), col="blue", lwd=2,
     main="ROC Curve for Models")
abline(0,1) 

#Naïve Bayes
library(e1071)
library(ROCR)
set.seed(32505485)

# - Classification Model
nbmodel <- naiveBayes(MHT ~., data = WAUS.train)

# - Accuracy & Confusion Matrix
nbpred <- predict(nbmodel, WAUS.test)
nbmatrix <- table(actual = WAUS.test$MHT, predicted = nbpred)
nbmatrix
nbaccuracy <- sum(diag(nbmatrix))/sum(nbmatrix)
nbaccuracy

# - AUC and ROC
nb.bayes = predict (nbmodel, WAUS.test, type = 'raw')
nbcpred <- prediction ( nb.bayes [,2], WAUS.test$MHT)
nbcperf <- performance (nbcpred, "tpr", "fpr")
plot (nbcperf, add=TRUE, col = "orange")
nb_auc <- performance(nbcpred, "auc")@y.values[[1]]
nb_auc


#Bagging
library(adabag)
library(rpart)
set.seed(32505485)

# - Classification Model
bagmodel <- bagging(MHT ~ ., data = WAUS.train)
# - Accuracy & Confusion Matrix
bagpred <- predict.bagging(bagmodel, newdata = WAUS.test)
bagcm <- table(actual = WAUS.test$MHT, predicted = bagpred$class)
bagcm
bagaccuracy <- sum(diag(bagcm)) / sum(bagcm)
bagaccuracy

# - AUC and ROC
bagpred_rocr <- prediction(bagpred$prob[,2], WAUS.test$MHT)
bagperf <- performance(bagpred_rocr, "tpr", "fpr")
plot(bagperf, add=TRUE, col = "red")
bagperf_auc <- performance(bagpred_rocr, "auc")
bagperf_auc <- bagperf_auc@y.values[[1]]
bagperf_auc

bagmodel$importance


#Boosting
library(adabag)
library(rpart)
set.seed(32505485)

# - Classification Model
boostmodel <- boosting(MHT ~ ., data = WAUS.train)

# - Accuracy & Confusion Matrix
boostpred <- predict.boosting(boostmodel, newdata = WAUS.test)
boostcm <- table(actual = WAUS.test$MHT, predicted = boostpred$class)
boostcm
boostaccuracy <- sum(diag(boostcm)) / sum(boostcm)
boostaccuracy

# - AUC and ROC
boostpred_rocr <- prediction(boostpred$prob[,2], WAUS.test$MHT)
boostperf <- performance(boostpred_rocr, "tpr", "fpr")
plot(boostperf, add=TRUE, col = "green")

boostperf_auc <- performance(boostpred_rocr, "auc")
boostperf_auc <- boostperf_auc@y.values[[1]]
boostperf_auc

#
boostmodel$importance



#Random Forest 
library(randomForest)
set.seed(32505485)

# - Classification Model
rfmodel <- randomForest(MHT ~ ., data = WAUS.train)
rfmodel

# - Accuracy & Confusion Matrix
rfpred <- predict(rfmodel, newdata = WAUS.test)
rfcm <- table(rfpred, WAUS.test$MHT)
rfcm
rfaccuracy <- sum(diag(rfcm)) / sum(rfcm)
rfaccuracy

# - AUC and ROC
rfpred.rf <- predict(rfmodel, newdata = WAUS.test, type = "prob")
rf_pred <- prediction(rfpred.rf[,2], WAUS.test$MHT)
rf_perf <- performance(rf_pred, "tpr", "fpr")
plot(rf_perf, add=TRUE, col = "purple")

rf_auc <- performance(rf_pred, "auc")@y.values[[1]]
rf_auc

var_importance <- importance(rfmodel)
var_importance

legend("bottomright", legend = c("Decision Tree", "Naïve Bayes", "Bagging", "Boosting", "Random Forest"),
       col = c("blue", "orange", "red", "green", "purple"), lwd = 2)


# - Simple Classifier 
library(tree)
library(ROCR)
set.seed(32505485)
simple_model <- tree(MHT ~ WindDir3pm + WindDir9am + WindGustDir, data = WAUS.train)
# Prune the decision tree to reduce depth
pruned_model <- prune.tree(simple_model, best = 7)  # Adjust the "best" parameter to control pruning
# Plot the pruned tree
plot(pruned_model)
text(pruned_model, pretty = 0)
simple_pred_model <- predict(pruned_model, newdata = WAUS.test, type = "class")
print(simple_pred_model)
table(simple_pred_model, WAUS.test$MHT)
accuracy <- sum(simple_pred_model == WAUS.test$MHT) / nrow(WAUS.test)
accuracy


simple_pred <- predict(simple_model, newdata = WAUS.test, type = "vector")
simple_confidence <- simple_pred[,2]
simple_pred_rocr <- prediction(simple_confidence, WAUS.test$MHT)
simple_auc <- performance(simple_pred_rocr, "auc")@y.values[[1]]
simple_auc
plot(performance(simple_pred_rocr, "tpr", "fpr"), col="red", lwd=2,
     main="Simple Decision Tree ROC Curve")
abline(0,1)



# Improved Classifier
#Decision Tree
library(tree)
library(ROCR)
library(caret)
set.seed(32505485)
tree_model <- tree(MHT ~ WindDir3pm + WindDir9am + WindGustDir + Sunshine  + Evaporation + MaxTemp + Pressure3pm +Pressure9am, data = WAUS.train)
summary(tree_model)
plot(tree_model)
text(tree_model, pretty = 0)
tree_pred_model <- predict(tree_model, newdata = WAUS.test, type = "class")
print(tree_pred_model)
table(tree_pred_model, WAUS.test$MHT)
accuracy <- sum(tree_pred_model == WAUS.test$MHT) / nrow(WAUS.test)
accuracy
tree_pred <- predict(tree_model, newdata = WAUS.test, type = "vector")
tree_confidence <- tree_pred[,2]
tree_pred_rocr <- prediction(tree_confidence, WAUS.test$MHT)
tree_auc <- performance(tree_pred_rocr, "auc")@y.values[[1]]
tree_auc
plot(performance(tree_pred_rocr, "tpr", "fpr"), col="blue", lwd=2,
     main="ROC Curve for Improved Decision Tree")
abline(0,1) 

library(tree)
library(ROCR)
set.seed(32505485)
tree_model <- tree(MHT ~ WindDir3pm + WindDir9am + WindGustDir + Sunshine  + Evaporation + MaxTemp + Pressure3pm +Pressure9am, data = WAUS.train)
text(tree_model, pretty = 0)
cv_results <- cv.tree(tree_model, FUN = prune.misclass)
print(cv_results)
pruned_tree <- prune.misclass(tree_model, best = cv_results$size[which.min(cv_results$dev)])
tree_pred <- predict(pruned_tree, newdata = WAUS.test, type = "vector")
print(tree_pred_model)
table(tree_pred_model, WAUS.test$MHT)
accuracy <- sum(tree_pred_model == WAUS.test$MHT) / nrow(WAUS.test)
accuracy
tree_confidence <- tree_pred[, 2]
tree_pred_rocr1 <- prediction(tree_confidence, WAUS.test$MHT)
tree_auc <- performance(tree_pred_rocr1, "auc")@y.values[[1]]
tree_auc
plot(performance(tree_pred_rocr1, "tpr", "fpr"), col="red", lwd=2,
     main="ROC Curve for Pruned and CV Model")
abline(0,1) 








library(neuralnet)
set.seed(32505485)
WAUS.train <- as.data.frame(WAUS.train)
# Convert factor variables to character type
factor_vars <- c("WindDir3pm", "WindDir9am", "WindGustDir", "RainToday", "Location")
WAUS.train[factor_vars] <- lapply(WAUS.train[factor_vars], as.character)
#model matrix
train_matrix <- model.matrix(MHT ~ . - 1, data = WAUS.train)
train_target <- WAUS.train$MHT
# neural network
neural_model <- neuralnet(MHT ~ ., data = cbind(train_matrix, MHT = train_target), hidden = 3)
# prediction 
WAUS.test_input <- WAUS.test[, !(colnames(WAUS.test) %in% c("MHT"))]
WAUS.test_input[factor_vars] <- lapply(WAUS.test_input[factor_vars], as.character)
test_matrix <- model.matrix(~ . - 1, data = WAUS.test_input)
predictions <- compute(neural_model, test_matrix)$net.result
# predictions to classes
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
actual_classes <- as.numeric(as.character(WAUS.test$MHT))
# Calculate accuracy & AUC
accuracy <- mean(predicted_classes == actual_classes)
accuracy
prediction_obj <- prediction(predictions, actual_classes)
# Calculate AUC
auc <- performance(prediction_obj, "auc")@y.values[[1]]
auc
# Plot neural network
plot(neural_model)
neural_model$result.matrix
print(neural_model$weights)



library(e1071)
library(ROCR)
set.seed(32505485)

svm_model <- svm(MHT ~ ., data = WAUS.train)

# Accuracy & Confusion Matrix
svm_pred <- predict(svm_model, newdata = WAUS.test)
svm_cm <- table(actual = WAUS.test$MHT, predicted = svm_pred)
svm_accuracy <- sum(diag(svm_cm)) / sum(svm_cm)
svm_accuracy

# AUC and ROC
svm_pred_rocr <- prediction(as.numeric(svm_pred), WAUS.test$MHT)
svm_perf <- performance(svm_pred_rocr, "tpr", "fpr")
plot(svm_perf, col = "green", main = "SVM ROC plot")
abline(0,1)

svm_perf_auc <- performance(svm_pred_rocr, "auc")
svm_perf_auc <- svm_perf_auc@y.values[[1]]
svm_perf_auc


































