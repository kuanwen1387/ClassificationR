#Loading library for SVM, Naive Bayes and ROC plot
library(e1071)
library(ROCR)
library(corrplot)
library(caret)

colNames = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

#Load iris data
iris.data = read.table("iris.data", header = FALSE, col.names = colNames)

#Convert Species to nominal
iris.data[ , 5] = factor(iris.data[ , 5])

#Summary of descriptive stats for iris.data
summary(iris.data)

#Print the mean of all atributes
print(paste("Mean of Sepal Length:", mean(iris.data$Sepal.Length)))
print(paste("Mean of Sepal Width:", mean(iris.data$Sepal.Width)))
print(paste("Mean of Petal Length:", mean(iris.data$Petal.Length)))
print(paste("Mean of Petal Width:", mean(iris.data$Petal.Width)))

#Covariance matrix for iris data
cov(iris.data[, -5])

#Correlation efficient of iris data
iris.data.cor = cor(iris.data[, -5])
corrplot(iris.data.cor, method = "number")

#Run experiments with various training/test sizes and save results
irisNB.bestModel = data.frame("Percent.Training" = double(), Accuracy = double(0))
percent = c(0.25, 0.5, 0.75)

for (percentIndex in 1:length(percent))
{
  for (index in 1:20)
  {#Split iris.data into training and test set
    indexes = sample(1:nrow(iris), size = percent[percentIndex] * nrow(iris.data))
    
    iris.data.training = iris.data[indexes, ]
    iris.data.test = iris.data[-indexes, ]
    
    #Train using naive bayes
    irisNB.model = naiveBayes(iris.data.training[,1:4], iris.data.training[,5])
    irisNB.predTable = table(predict(irisNB.model, iris.data.test[,-5]), iris.data.test[,5])
    irisNB.conMatrix = confusionMatrix(irisNB.predTable)
    irisNB.bestModel[nrow(irisNB.bestModel) + 1, ] = c(percent[percentIndex], irisNB.conMatrix$overall['Accuracy'])
  }
}

irisNB.bestModel

irisNB.avg.acc = data.frame("Percent.Training" = double(), Accuracy = double(0))

#Compute average accuracy for all proportions
for (percentIndex in 1:length(percent))
{
  temp = c(percent[percentIndex] * 100, 0)
  
  for (accIndex in 1:nrow(irisNB.bestModel))
  {
    if (irisNB.bestModel[accIndex, 1] == percent[percentIndex])
    {
      temp[2] = temp[2] + irisNB.bestModel[accIndex, 2]
    }
  }
  temp[2] = temp[2] / 20
  irisNB.avg.acc[nrow(irisNB.avg.acc) + 1, ] = temp
}

#Print average accuracy from experiements
irisNB.avg.acc

for (index in 1:nrow(irisNB.avg.acc))
{
  print(paste0("Mean accuracy for ", irisNB.avg.acc[index, 1], "% training set is ", irisNB.avg.acc[index, 2] * 100, "%."))
}

#Plot for training set proportion and accuracy
plot(irisNB.avg.acc[, 1], irisNB.avg.acc[, 2], type = "o", col = "red", main="Training Set Proportion and Accuracy (Naive Bayes: Iris)", xlab="Training Set %", ylab="Accuracy %", pch = 19)

irisNB.maxIndex = which.max(irisNB.avg.acc[, 2])

irisNB.max = 0

#Get the best proportion for training set
for (accIndex in 1:nrow(irisNB.bestModel))
{
  if (irisNB.bestModel[accIndex, 1] == percent[irisNB.maxIndex] && irisNB.bestModel[accIndex, 2] > irisNB.max)
    irisNB.max = irisNB.bestModel[accIndex, 2]
}

#Print highest accuracy for best proportion
print(paste0("Highest accuracy achieved with ", percent[irisNB.maxIndex] * 100, "% training set at ", irisNB.max * 100, "% accuracy."))

#Print confusion matrix for all proportions
for (percentIndex in 1:length(percent))
{#Split iris data into training and test set
  indexes = sample(1:nrow(iris), size = percent[percentIndex] * nrow(iris.data))
  
  iris.data.training = iris.data[indexes, ]
  iris.data.test = iris.data[-indexes, ]
  
  irisNB.model = naiveBayes(iris.data.training[,1:4], iris.data.training[,5])
  irisNB.predTable = table(predict(irisNB.model, iris.data.test[,-5]), iris.data.test[,5])
  irisNB.conMatrix = confusionMatrix(irisNB.predTable)
  print(paste(percent[percentIndex] * 100, "% Training set with", irisNB.conMatrix$overall['Accuracy'] * 100, "% accuracy"))
  print(irisNB.predTable)
}

#Run experiments removing one feature each experiment
irisNB.bestModel = data.frame(Feature = integer(0), Accuracy = double(0))

#Remove feature and run 20 experiments
for (featureIndex in 1:(length(iris.data) - 1))
{
  for (index in 1:20)
  {#Remove one attribute and run 20 experiments
    iris.data.feature = iris.data[, -featureIndex]
    
    indexes = sample(1:nrow(iris), size = percent[irisNB.maxIndex] * nrow(iris.data.feature))
    
    iris.data.training = iris.data.feature[indexes, ]
    iris.data.test = iris.data.feature[-indexes, ]
    
    irisNB.model = naiveBayes(iris.data.training[,1:3], iris.data.training[,4])
    irisNB.predTable = table(predict(irisNB.model, iris.data.test[,-4]), iris.data.test[,4])
    irisNB.conMatrix = confusionMatrix(irisNB.predTable)
    irisNB.bestModel[nrow(irisNB.bestModel) + 1, ] = c(featureIndex, irisNB.conMatrix$overall['Accuracy'])
  }
}

#Output results
irisNB.bestModel

irisNB.avg.acc = data.frame(Feature = integer(), Accuracy = double(0))

#Compute average accuracy after feature removal
for (featureIndex in 1:(length(iris.data) - 1))
{
  temp = c(featureIndex, 0)
  
  for (accIndex in 1:nrow(irisNB.bestModel))
  {
    if (irisNB.bestModel[accIndex, 1] == featureIndex)
    {
      temp[2] = temp[2] + irisNB.bestModel[accIndex, 2]
    }
  }
  temp[2] = temp[2] / 20
  irisNB.avg.acc[nrow(irisNB.avg.acc) + 1, ] = temp
}

#Print average accuracy from experiements
irisNB.avg.acc

for (index in 1:nrow(irisNB.avg.acc))
{
  print(paste0("Mean accuracy for removal of attribute ", irisNB.avg.acc[index, 1], " is ", irisNB.avg.acc[index, 2] * 100, "%."))
}

irisNB.avg.acc[, 1] = factor(irisNB.avg.acc[, 1])

#Plot for feature removal and accuracy
plot(irisNB.avg.acc[, 1], irisNB.avg.acc[, 2], main="Feature Removal and Accuracy (Naive Bayes: Iris)", xlab="Training Set %", ylab="Accuracy %", pch = 19)

irisNB.maxIndex = which.max(irisNB.avg.acc[, 2])

irisNB.max = 0

#Get the best proportion feature removal
for (accIndex in 1:nrow(irisNB.bestModel))
{
  if (irisNB.bestModel[accIndex, 1] == irisNB.maxIndex && irisNB.bestModel[accIndex, 2] > irisNB.max)
    irisNB.max = irisNB.bestModel[accIndex, 2]
}

#Print highest accuracy for feature removal
print(paste0("Highest accuracy achieved removing attribute ", irisNB.maxIndex, " at ", irisNB.max * 100, "% accuracy."))


#Print confusion matrix for feature removal
for (featureIndex in 1:(length(iris.data) - 1))
{
  iris.data.feature = iris.data[, -featureIndex]
  
  indexes = sample(1:nrow(iris), size = percent[irisNB.maxIndex] * nrow(iris.data.feature))
  
  iris.data.training = iris.data.feature[indexes, ]
  iris.data.test = iris.data.feature[-indexes, ]
  
  irisNB.model = naiveBayes(iris.data.training[,1:3], iris.data.training[,4])
  irisNB.predTable = table(predict(irisNB.model, iris.data.test[,-4]), iris.data.test[,4])
  irisNB.conMatrix = confusionMatrix(irisNB.predTable)
  print(paste("Removed ", colnames(iris.data)[featureIndex], "with", irisNB.conMatrix$overall['Accuracy'] * 100, "% accuracy"))
  print(irisNB.predTable)
}


#Task 2
#Set header for aus data
colNames = c("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "Class")

#Load australian data
ausCredit.data = read.table("australian.data", col.names = colNames)

#preprocess nominal attributes
ausCredit.data[, 1] = factor(ausCredit.data[, 1])
ausCredit.data[, 4] = factor(ausCredit.data[, 4])
ausCredit.data[, 5] = factor(ausCredit.data[, 5])
ausCredit.data[, 6] = factor(ausCredit.data[, 6])
ausCredit.data[, 8] = factor(ausCredit.data[, 8])
ausCredit.data[, 9] = factor(ausCredit.data[, 9])
ausCredit.data[, 11] = factor(ausCredit.data[, 11])
ausCredit.data[, 12] = factor(ausCredit.data[, 12])
ausCredit.data[, 15] = factor(ausCredit.data[, 15])

#Summary descriptive stats for ausCredit.data
summary(ausCredit.data)

#Print variance no numeric attributes
print(paste("Variance of A2:", var(ausCredit.data[, 2])))
print(paste("Variance of A3:", var(ausCredit.data[, 3])))
print(paste("Variance of A7:", var(ausCredit.data[, 7])))
print(paste("Variance of A10:", var(ausCredit.data[, 10])))
print(paste("Variance of A13:", var(ausCredit.data[, 13])))
print(paste("Variance of A14:", var(ausCredit.data[, 14])))

#Mosai plot for selected A4 and A12
mosaicplot(ausCredit.data$Class~ausCredit.data$A4, col = c("red", "orange", "yellow", "green", "blue", "purple"), main = "Mosaic Plot Class and A4 (AusData)", xlab = "Class", ylab = "A14")
mosaicplot(ausCredit.data$Class~ausCredit.data$A12, col = c("red", "orange", "yellow", "green", "blue", "purple"), main = "Mosaic Plot Class and A12 (AusData)", xlab = "Class", ylab = "A12")

#Run experiments with various training/test sizes
ausNB.bestModel = data.frame("Percent.Training" = double(), Accuracy = double(0))

for (percentIndex in 1:length(percent))
{
  for (index in 1:20)
  {#Split ausCredit.data into training and test set
    indexes = sample(1:nrow(ausCredit.data), size = percent[percentIndex] * nrow(ausCredit.data))
    
    ausCredit.data.training = ausCredit.data[indexes, ]
    ausCredit.data.test = ausCredit.data[-indexes, ]
    
    #Train ausCredit with naive bayes
    ausNB.model = naiveBayes(ausCredit.data.training[, 1:14], factor(ausCredit.data.training[, 15]))
    ausNB.predTable = table(predict(ausNB.model, ausCredit.data.test[, -15]), factor(ausCredit.data.test[, 15]))
    ausNB.conMatrix = confusionMatrix(ausNB.predTable)
    ausNB.bestModel[nrow(ausNB.bestModel) + 1, ] = c(percent[percentIndex], ausNB.conMatrix$overall['Accuracy'])
  }
}

ausNB.bestModel

ausNB.avg.acc = data.frame("Percent.Training" = double(), Accuracy = double(0))

#Compute average accuracy for each proportion
for (percentIndex in 1:length(percent))
{
  #ausNB.avg.acc[percentIndex] = 0;
  temp = c(percent[percentIndex] * 100, 0)
  
  for (accIndex in 1:nrow(ausNB.bestModel))
  {
    if (ausNB.bestModel[accIndex, 1] == percent[percentIndex])
    {
      #ausNB.avg.acc[percentIndex] = ausNB.avg.acc[percentIndex] + ausNB.bestModel[accIndex, 2]
      temp[2] = temp[2] + ausNB.bestModel[accIndex, 2]
    }
  }
  #ausNB.avg.acc[percentIndex] = ausNB.avg.acc[percentIndex] / 20
  temp[2] = temp[2] / 20
  ausNB.avg.acc[nrow(ausNB.avg.acc) + 1, ] = temp
}

#Print average accuracy from experiements
ausNB.avg.acc

for (index in 1:nrow(ausNB.avg.acc))
{
  print(paste0("Mean accuracy for ", ausNB.avg.acc[index, 1], "% training set is ", ausNB.avg.acc[index, 2] * 100, "%."))
}

#Plot for training set proportion and accuracy
plot(ausNB.avg.acc[, 1], ausNB.avg.acc[, 2], type = "o", col = "red", main="Training Set Proportion and Accuracy (Naive Bayes: AusCredit)", xlab="Training Set %", ylab="Accuracy %", pch = 19)

ausNB.maxIndex = which.max(ausNB.avg.acc[, 2])

ausNB.max = 0

#Get the best proportion for training set
for (accIndex in 1:nrow(ausNB.bestModel))
{
  if (ausNB.bestModel[accIndex, 1] == percent[ausNB.maxIndex] && ausNB.bestModel[accIndex, 2] > ausNB.max)
    ausNB.max = ausNB.bestModel[accIndex, 2]
}

#Print highest accuracy for best proportion
print(paste0("Highest accuracy achieved with ", percent[ausNB.maxIndex] * 100, "% training set at ", ausNB.max * 100, "% accuracy."))

#Print confusion matrix for all proportions
for (percentIndex in 1:length(percent))
{#Split ausCredit.data into training and test set
  indexes = sample(1:nrow(ausCredit.data), size = percent[percentIndex] * nrow(ausCredit.data))
    
  ausCredit.data.training = ausCredit.data[indexes, ]
  ausCredit.data.test = ausCredit.data[-indexes, ]
  
  ausNB.model = naiveBayes(ausCredit.data.training[, 1:14], factor(ausCredit.data.training[, 15]))
  ausNB.predTable = table(predict(ausNB.model, ausCredit.data.test[, -15]), factor(ausCredit.data.test[, 15]))
  ausNB.conMatrix = confusionMatrix(ausNB.predTable)
  print(paste(percent[percentIndex] * 100, "% Training set with", ausNB.conMatrix$overall['Accuracy'] * 100, "% accuracy"))
  print(ausNB.predTable)
}

#Set gamma and cost range for grid search
costRange = c(-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15)
gammaRange = c(-15, -13, -11, -9, -7, -5, -3, -1, 1, 3)

#Get best cost and gamma
bestPrameter = tune(svm, Class~., data = ausCredit.data, ranges = list(cost = 2^costRange, gamma = 2^gammaRange))

bestPrameter

#Run experiments with various training/test sizes using best cost and gamma
ausSVM.bestModel = data.frame("Percent.Training" = double(), Accuracy = double(0))

#Run svm on various training set proportions
for (percentIndex in 1:length(percent))
{
  for (index in 1:20)
  {#Split ausCredit.data into training and test set
    indexes = sample(1:nrow(ausCredit.data), size = percent[percentIndex] * nrow(ausCredit.data))
    
    ausCredit.data.training = ausCredit.data[indexes, ]
    ausCredit.data.test = ausCredit.data[-indexes, ]
    
    ausSVM.model = svm(Class~., data = ausCredit.data.training, cost = 2, gamma = 0.03125)
    ausSVM.predTable = table(prediction = predict(ausSVM.model, ausCredit.data.test[, -15]), truth = ausCredit.data.test[, 15])
    ausSVM.conMatrix = confusionMatrix(ausSVM.predTable)
    ausSVM.bestModel[nrow(ausSVM.bestModel) + 1, ] = c(percent[percentIndex], ausSVM.conMatrix$overall['Accuracy'])
  }
}

ausSVM.bestModel

ausSVM.avg.acc = data.frame("Percent.Training" = double(), Accuracy = double(0))

#Compute average accuracy for each proportion
for (percentIndex in 1:length(percent))
{
  #ausSVM.avg.acc[percentIndex] = 0;
  temp = c(percent[percentIndex] * 100, 0)
  
  for (accIndex in 1:nrow(ausSVM.bestModel))
  {
    if (ausSVM.bestModel[accIndex, 1] == percent[percentIndex])
    {
      #ausSVM.avg.acc[percentIndex] = ausSVM.avg.acc[percentIndex] + ausSVM.bestModel[accIndex, 2]
      temp[2] = temp[2] + ausSVM.bestModel[accIndex, 2]
    }
  }
  #ausSVM.avg.acc[percentIndex] = ausSVM.avg.acc[percentIndex] / 20
  temp[2] = temp[2] / 20
  ausSVM.avg.acc[nrow(ausSVM.avg.acc) + 1, ] = temp
}

#Print average accuracy from experiements
ausSVM.avg.acc

for (index in 1:nrow(ausSVM.avg.acc))
{
  print(paste0("Mean accuracy for ", ausSVM.avg.acc[index, 1], "% training set on SVM is ", ausSVM.avg.acc[index, 2] * 100, "%."))
}

#Plot for training set proportion and accuracy
plot(ausSVM.avg.acc[, 1], ausSVM.avg.acc[, 2], type = "o", col = "red", main="Training Set Proportion and Accuracy (SVM: AusCredit)", xlab="Training Set %", ylab="Accuracy %", pch = 19)

ausSVM.maxIndex = which.max(ausSVM.avg.acc[, 2])

ausSVM.max = 0

#Get the best proportion for training set
for (accIndex in 1:nrow(ausSVM.bestModel))
{
  if (ausSVM.bestModel[accIndex, 1] == percent[ausSVM.maxIndex] && ausSVM.bestModel[accIndex, 2] > ausSVM.max)
    ausSVM.max = ausSVM.bestModel[accIndex, 2]
}

#Print highest accuracy for best proportion
print(paste0("Highest accuracy achieved with ", percent[ausSVM.maxIndex] * 100, "% training set at ", ausSVM.max * 100, "% accuracy for SVM."))

#Print confusion matrix for all proportions
for (percentIndex in 1:length(percent))
{
  indexes = sample(1:nrow(ausCredit.data), size = percent[percentIndex] * nrow(ausCredit.data))
  
  ausCredit.data.training = ausCredit.data[indexes, ]
  ausCredit.data.test = ausCredit.data[-indexes, ]
  
  #Get probabilities for ROC
  ausSVM.model = svm(Class~., data = ausCredit.data.training, cost = 2, gamma = 0.03125, probability = TRUE)
  ausSVM.prob = predict(ausSVM.model, ausCredit.data.test[, -15], probability = TRUE, decision.value = TRUE)
  ausSVM.predTable = table(prediction = predict(ausSVM.model, ausCredit.data.test[, -15]), truth = ausCredit.data.test[, 15])
  ausSVM.conMatrix = confusionMatrix(ausSVM.predTable)
  print(paste(percent[percentIndex] * 100, "% Training set with", ausSVM.conMatrix$overall['Accuracy'] * 100, "% accuracy for SVM."))
  print(ausSVM.predTable)
  
  #Plot ROC graph
  ausSNM.roc = prediction(attributes(ausSVM.prob)$decision.values, ausCredit.data.test[, 15])
  ausSVM.auc = performance(ausSNM.roc, 'tpr', 'fpr')
  plot(ausSVM.auc, main = paste(percent[percentIndex] * 100, "% Training Set"))
}

#Run experiments removing feature using best cost and gamma
ausSVM.bestModel = data.frame("Feature" = integer(), Accuracy = double(0))

#Feature to remove
selectFeature = c(4, 12)

#Run svm on after removing selected feature
for (featureIndex in 1:length(selectFeature))
{
  for (index in 1:20)
  {
    #Remove one attribute and run 20 experiments
    ausCredit.data.feature = ausCredit.data[, -selectFeature[featureIndex]]
    
    #Split ausCredit.data into training and test set
    indexes = sample(1:nrow(ausCredit.data), size = percent[ausSVM.maxIndex] * nrow(ausCredit.data.feature))
    
    ausCredit.feature.data.training = ausCredit.data.feature[indexes, ]
    ausCredit.feature.data.test = ausCredit.data.feature[-indexes, ]
    
    ausSVM.model = svm(Class~., data = ausCredit.feature.data.training, cost = 2, gamma = 0.03125)
    ausSVM.predTable = table(prediction = predict(ausSVM.model, ausCredit.feature.data.test[, -14]), truth = ausCredit.feature.data.test[, 14])
    ausSVM.conMatrix = confusionMatrix(ausSVM.predTable)
    ausSVM.bestModel[nrow(ausSVM.bestModel) + 1, ] = c(selectFeature[featureIndex], ausSVM.conMatrix$overall['Accuracy'])
  }
}

ausSVM.bestModel

ausSVM.avg.acc = data.frame(Feature = integer(), Accuracy = double(0))

#Compute average accuracy for each feature removal
for (featureIndex in 1:length(selectFeature))
{
  temp = c(selectFeature[featureIndex], 0)
  
  for (accIndex in 1:nrow(ausSVM.bestModel))
  {
    if (ausSVM.bestModel[accIndex, 1] == selectFeature[featureIndex])
    {
      temp[2] = temp[2] + ausSVM.bestModel[accIndex, 2]
    }
  }
  temp[2] = temp[2] / 20
  ausSVM.avg.acc[nrow(ausSVM.avg.acc) + 1, ] = temp
}

#Print average accuracy from experiements
ausSVM.avg.acc

#Print confusion matrix after removing selected feature
for (featureIndex in 1:length(selectFeature))
{
    #Remove one attribute and run 20 experiments
    ausCredit.data.feature = ausCredit.data[, -selectFeature[featureIndex]]
    
    #Split ausCredit.data into training and test set
    indexes = sample(1:nrow(ausCredit.data), size = percent[ausSVM.maxIndex] * nrow(ausCredit.data.feature))
    
    ausCredit.feature.data.training = ausCredit.data.feature[indexes, ]
    ausCredit.feature.data.test = ausCredit.data.feature[-indexes, ]
    
    ausSVM.model = svm(Class~., data = ausCredit.feature.data.training, cost = 2, gamma = 0.03125)
    ausSVM.predTable = table(prediction = predict(ausSVM.model, ausCredit.feature.data.test[, -14]), truth = ausCredit.feature.data.test[, 14])
    ausSVM.conMatrix = confusionMatrix(ausSVM.predTable)
    ausSVM.bestModel[nrow(ausSVM.bestModel) + 1, ] = c(selectFeature[featureIndex], ausSVM.conMatrix$overall['Accuracy'])
    print(ausSVM.predTable)
}