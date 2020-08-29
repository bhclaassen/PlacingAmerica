##############################
#  Read in and organize data #
##############################

############################################################
# Functions #

#Function: Rotate image
rotate <- function(x) t(apply(x, 2, rev))
#Function: Special image function for correct digit orientation
img <- function(x) image(rotate(t(x)), col = grey(seq(0, 1, length = 256)))


############################################################

#Function: Read in training data
csvToArray <- function(tmpCsvName)
{
  #Read in CSV
  tmpCSV <- read.csv(tmpCsvName, stringsAsFactors = F, header = F)
  
  #Initialize storage array
  tmpArray <- array(, c(16,16,dim(tmpCSV)[1]) )
  
  #Iterate across every row in CSV
  for(i in 1:dim(tmpCSV)[1])
  {
    #Assign row to the i-th 16x16 matrix
    tmpArray[,,i] <- c(t(as.data.frame(tmpCSV[i,])))
    print(i)
  }
  
  #Print first image
  img(tmpArray[,,1])
  
  #Print dimensions
  print(dim(tmpArray))
  return(tmpArray)
}


############################################################

#Function: Read in training data
csvToArray_Test <- function(tmpCsvName)
{
  #Read in CSV
  tmpCSV <- read.csv(tmpCsvName, stringsAsFactors = F, header = F)
  
  #Initialize storage array
  tmpArray <- array(, c(16,16,dim(tmpCSV)[1]) )
  
  
  #Iterate across every row in CSV
  for(i in 1:dim(tmpCSV)[1])
  {
    #Assign row to the i-th 16x16 matrix
    tmpArray[,,i] <- c(t(as.data.frame(tmpCSV[i,-1]))) #First column is actual digit class
    print(i)
  }
  
  #Print first image
  img(tmpArray[,,1])
  
  #Print dimensions
  print(dim(tmpArray))
  return(tmpArray)
}

############################################################
#Function: Count the number of modes in a density function
findModes<- function(tmpDensity)
{
  modes <- NULL
  for(i in 2:(length(tmpDensity)-1))
  {
    if( (tmpDensity[i] > tmpDensity[i-1]) & (tmpDensity[i] > tmpDensity[i+1]) )
    {
      modes <- c(modes,i)
    }
  }
  if(length(modes) == 0)
  {
    modes = 'Monotonic Distribution'
  }
  return(modes)
}

############################################################
#Function: Find k-nearest neighbors
nearestNeighbors <- function(testData)
{
  #Extract row- and column-wise feature data
  #rm(features)
  setwd("~/Documents/_Documents/Reno/Reno_2019-Spring/MachineLearning/Homework1/Features")
  load("Features.R")
  
  #Initialize storage matrix for test point predictions (Set to '-1')
  tmpTestPredictions <- rep(-1, dim(testData)[3])
  
  #Iterate across all test points
  for(t in 1:dim(testData)[3])
  {
    print(t)
    #Initialize temporary density storage info
    tmpDensity <- rep(NA, 16)
      
    #Round data to one digit
    currentObs <- round(testData[,,t], 1)
    
    #Extract row-wise and column-wise sum densities
    tmpRowSumDensity <- density(rowSums(currentObs))$y
    tmpColSumDensity <- density(colSums(currentObs))$y
    
    #Extract summary statistics, std dev, and number of modes from density of row-wise sums
    tmpDensity[1] <- min(tmpRowSumDensity)                 #Min
    tmpDensity[2] <- quantile(tmpRowSumDensity, .25)       #1st quartile
    tmpDensity[3] <- quantile(tmpRowSumDensity, .50)       #Median
    tmpDensity[4] <- mean(tmpRowSumDensity)                #Mean
    tmpDensity[5] <- quantile(tmpRowSumDensity, .75)       #3rd quartile
    tmpDensity[6] <- max(tmpRowSumDensity)                 #Max
    tmpDensity[7] <- sd(tmpRowSumDensity)                  #Std dev
    tmpDensity[8] <- length(findModes(tmpRowSumDensity))  #Number of modes
    
    #Extract summary statistics, std dev, and number of modes from density of column-wise sums
    tmpDensity[9] <- min(tmpColSumDensity)                  #Min
    tmpDensity[10] <- quantile(tmpColSumDensity, .25)       #1st quartile
    tmpDensity[11] <- quantile(tmpColSumDensity, .50)       #Median
    tmpDensity[12] <- mean(tmpColSumDensity)                #Mean
    tmpDensity[13] <- quantile(tmpColSumDensity, .75)       #3rd quartile
    tmpDensity[14] <- max(tmpColSumDensity)                 #Max
    tmpDensity[15] <- sd(tmpColSumDensity)                  #Std dev
    tmpDensity[16] <- length(findModes(tmpColSumDensity))  #Number of modes
    
    
    #Calculate distance between test point info and EACH other point
    testDistances <- calculateDistances(tmpDensity)

    
    if(t == 1)
    {
      allDistances <- testDistances
    } else
    {
      allDistances <- cbind(allDistances,testDistances[,2])
    }
  }
  
  
  
  #Return predictions
  return(allDistances)
}

#rm(testData, tmpTestPredictions, tmpDensity, tmpRowSumDensity, tmpColSumDensity, currentObs)
############################################################
#Function: Calculate distance
calculateDistances <- function(testDensity)
{
  #Initialize distance storage
  tmpDistances <- as.data.frame(matrix(,dim(features)[1],2))
  names(tmpDistances) <- c("Digit", "Distance")
  
  #Iterate across all points in 'features'
  for(f in 1:dim(features)[1])
  {
    tmpFeature <- features[f, -c(1:2)]
    
    tmpDistances[f,1] <- features[f,1]
    tmpDistances[f,2] <- sum((tmpFeature-testDensity)^2)
  }
  
  #Return distance list
  return(tmpDistances)
}

#rm(testDensity, tmpDistances, tmpFeature)

############################################################
kNN <- function(tmpDistances, kNeighbors)
{
  #Initialize all predicitons storage
  allPredictions <- NULL
  
  #Iterate across distances for each test point
  for(p in 2:dim(tmpDistances)[2]) #Start after digit trained on in first column
  {
    #Sort list from smallest distance
    currentDistances <- tmpDistances[order(tmpDistances[,p]), c(1,p)]
    
    #Return most common digit from 'kNeighbors' closest points
    tmpPrediction <- as.numeric(names(which.max(table(currentDistances[1:kNeighbors,1]))))
    allPredictions <- c(allPredictions,tmpPrediction)
  }
  
  return(allPredictions)
}

#rm(tmpDistances, tmpPrediction, allPredictions, currentDistances)
############################################################
#Function: Calculate MSE for predicted digits
percSuccess <- function(tmpPredictions)
{
  #Initialize all predictions as failures
  tmpPredictions[,3] <- 1
  
  #Iterate across all predictions
  for(i in 1:dim(tmpPredictions)[1])
  {
    #If prediction is correct, change to success
    if(tmpPredictions[i,1] == tmpPredictions[i,2])
    {
      tmpPredictions[i,3] = 0
    }
  }
    
  #Calculate average success rate
  tmpSuccessRate <- sum( tmpPredictions[,3]  * 1/dim(tmpPredictions)[1] )
  return(tmpSuccessRate)
}
#rm(tmpPredictions)
############################################################
##################
# Begin Analysis #
##################
setwd("~/Documents/_Documents/Reno/Reno_2019-Spring/MachineLearning/Homework1/Data")
#Read in data
zeros <- csvToArray("Zeros.csv")
ones <- csvToArray("Ones.csv")
twos <- csvToArray("Twos.csv")
threes <- csvToArray("Threes.csv")
fours <- csvToArray("Fours.csv")
fives <- csvToArray("Fives.csv")
sixes <- csvToArray("Sixes.csv")
sevens <- csvToArray("Sevens.csv")
eights <- csvToArray("Eights.csv")
nines <- csvToArray("Nines.csv")

##################################################################

#####################
# Generate Features #
#####################

#Collate digits
digitsList <- list(ones,twos,threes,fours,fives,sixes,sevens,eights,nines,zeros)
#Initialize storage data frames
rowSumDat <- as.data.frame(matrix(,7291,10)) #7291 total training digits x 10 feature variables
names(rowSumDat) <- c("Digit","DigitCount","Min","FirstQ","Median","Mean","ThirdQ","Max","StdDev","NumModes")
colSumDat <- as.data.frame(matrix(,7291,10))
names(colSumDat) <- c("Digit","DigitCount","Min","FirstQ","Median","Mean","ThirdQ","Max","StdDev","NumModes")

for(d in 1:10) #Iterate across all digits and observations
{
  #Extract 'currentDigitDat'
  currentDigitDat <- digitsList[[d]]
  #Set number of obs in 'currentDigitDat'
  tmpN <- dim(currentDigitDat)[3]
  #Set start row for 'currentDigitDat' analysis (first NA)
  startRow <- which(is.na(rowSumDat[,1]))[1]
  
  #Iterate across all observations within 'currentDigitDat'
  for(o in startRow:(startRow+tmpN - 1)) #Five obs from tenth row: [10:(10 + 5 - 1)] = [10:15]
  {
    #Set 'Digit' col for storage matrices
    if(d == 10)
    {
      colSumDat[o,1] <- 0
      rowSumDat[o,1] <- 0  
    } else
    {
      colSumDat[o,1] <- d
      rowSumDat[o,1] <- d
    }
    
    #Set 'DigitCount' col for storage matrices
    colSumDat[o,2] <- o - startRow + 1 #Current row - 'startRow' + 1
    rowSumDat[o,2] <- o - startRow + 1 #Current row - 'startRow' + 1
    
    #Round data to one digit
    currentDigitObs <- currentDigitDat[,,(o - startRow + 1)] #Current row - 'startRow' + 1
    currentDigitObs <- round(currentDigitObs, 1)
    
    #Extract row-wise and column-wise sum densities
    tmpRowSumDensity <- density(rowSums(currentDigitObs))$y
    tmpColSumDensity <- density(colSums(currentDigitObs))$y
    
    #Extract summary statistics, std dev, and number of modes from density of row-wise sums
    rowSumDat[o,3] <- min(tmpRowSumDensity)                 #Min
    rowSumDat[o,4] <- quantile(tmpRowSumDensity, .25)       #1st quartile
    rowSumDat[o,5] <- quantile(tmpRowSumDensity, .50)       #Median
    rowSumDat[o,6] <- mean(tmpRowSumDensity)                #Mean
    rowSumDat[o,7] <- quantile(tmpRowSumDensity, .75)       #3rd quartile
    rowSumDat[o,8] <- max(tmpRowSumDensity)                 #Max
    rowSumDat[o,9] <- sd(tmpRowSumDensity)                  #Std dev
    rowSumDat[o,10] <- length(findModes(tmpRowSumDensity))  #Number of modes
    
    #Extract summary statistics, std dev, and number of modes from density of column-wise sums
    colSumDat[o,3] <- min(tmpColSumDensity)                 #Min
    colSumDat[o,4] <- quantile(tmpColSumDensity, .25)       #1st quartile
    colSumDat[o,5] <- quantile(tmpColSumDensity, .50)       #Median
    colSumDat[o,6] <- mean(tmpColSumDensity)                #Mean
    colSumDat[o,7] <- quantile(tmpColSumDensity, .75)       #3rd quartile
    colSumDat[o,8] <- max(tmpColSumDensity)                 #Max
    colSumDat[o,9] <- sd(tmpColSumDensity)                  #Std dev
    colSumDat[o,10] <- length(findModes(tmpColSumDensity))  #Number of modes
    
  }
  
  if(d == 10)
  {
    print("Number of NA's in whole storage matrix:")
    print(length(which(is.na(rowSumDat))))
    print(length(which(is.na(colSumDat))))
  }
}

#Delete variables created in checking above loop
# rm(d,o,currentDigitDat,tmpN,startRow,currentDigitObs, tmpRowSumDensity, tmpColSumDensity)

#Collate row- and column-wise sum features
features <- cbind(rowSumDat,colSumDat[, -c(1:2)])
names(features)[1:2] <- names(rowSumDat)[1:2]
names(features)[3:10] <- paste0("Row_", c("Min","FirstQ","Median","Mean","ThirdQ","Max","StdDev","NumModes"))
names(features)[11:18] <- paste0("Col_", c("Min","FirstQ","Median","Mean","ThirdQ","Max","StdDev","NumModes"))


#Write output to CSV
# setwd("~/Documents/_Documents/Reno/Reno_2019-Spring/MachineLearning/Homework1/Features")
# write.csv(rowSumDat, "RowSumDensities.csv", row.names = F)
# write.csv(colSumDat, "ColumnSumDensities.csv", row.names = F)
# write.csv(features, "Features.csv", row.names = F)
#
# save(rowSumDat, file = "RowSumDensities.R")
# save(colSumDat, file = "ColumnSumDensities.R")
# save(features, file = "Features.R")


#length(which(is.na(rowSumDat)))
#length(which(is.na(colSumDat)))

#colSumDat[1000:1010,1:5]
#rowSumDat[1000:1010,1:5]

###############################################################
######################
# Classify Test Data #
######################

#Read in test data
setwd("~/Documents/_Documents/Reno/Reno_2019-Spring/MachineLearning/Homework1/Data")
test <- csvToArray_Test("_TestData.csv")
#Read in key to test values
tmpTestFile <- read.csv("_TestData.csv", stringsAsFactors = F, header = F)
test_actualValues <- tmpTestFile[,1]
rm(tmpTestFile)


#Run Nearest Neighbors analysis
neighbors <- nearestNeighbors(test) #Outputs distances to each training point
length(which(is.na(neighbors)))

#Initialize classification distance storage matrices
k_1 <- as.data.frame(matrix(,893,2))
k_2 <- as.data.frame(matrix(,893,2))
k_3 <- as.data.frame(matrix(,893,2))
k_4 <- as.data.frame(matrix(,893,2))
k_5 <- as.data.frame(matrix(,893,2))
k_10 <- as.data.frame(matrix(,893,2))
k_15 <- as.data.frame(matrix(,893,2))
k_20 <- as.data.frame(matrix(,893,2))

names(k_1) <- c("Actual","Predicted_Mode")
names(k_2) <- c("Actual","Predicted_Mode")
names(k_3) <- c("Actual","Predicted_Mode")
names(k_4) <- c("Actual","Predicted_Mode")
names(k_5) <- c("Actual","Predicted_Mode")
names(k_10) <- c("Actual","Predicted_Mode")
names(k_15) <- c("Actual","Predicted_Mode")
names(k_20) <- c("Actual","Predicted_Mode")


tmpValues <- test_actualValues[1:893]
#Set actual values to first column
k_1[,1] <- tmpValues
k_2[,1] <- tmpValues
k_3[,1] <- tmpValues
k_4[,1] <- tmpValues
k_5[,1] <- tmpValues
k_10[,1] <- tmpValues
k_15[,1] <- tmpValues
k_20[,1] <- tmpValues


#Iterate across given values of k
k_1[,2] <- kNN(foo, 1)
k_2[,2] <- kNN(foo, 2)
k_3[,2] <- kNN(foo, 3)
k_4[,2] <- kNN(foo, 4)
k_5[,2] <- kNN(foo, 5)
k_10[,2] <- kNN(foo, 10)
k_15[,2] <- kNN(foo, 15)
k_20[,2] <- kNN(foo, 20)

#Calculate MSE
k_1_percSuccesses <- percSuccess(k_1)
k_2_percSuccesses <- percSuccess(k_2)
k_3_percSuccesses <- percSuccess(k_3)
k_4_percSuccesses <- percSuccess(k_4)
k_5_percSuccesses <- percSuccess(k_5)
k_10_percSuccesses <- percSuccess(k_10)
k_15_percSuccesses <- percSuccess(k_15)
k_20_percSuccesses <- percSuccess(k_20)

k_1_percSuccesses
k_2_percSuccesses
k_3_percSuccesses
k_5_percSuccesses
k_10_percSuccesses
k_15_percSuccesses
k_20_percSuccesses

successes <- c(k_1_percSuccesses, k_2_percSuccesses, k_3_percSuccesses, k_5_percSuccesses, k_10_percSuccesses, k_15_percSuccesses, k_20_percSuccesses)
#png("KNN.png")
plot(c(1,2,3,5,10,15,20), successes, type = 'l', ylab = "% Correct Predictions", xlab = "K Neighbors", ylim = c(0.75,1), main = "Prediction Success Rate by K Neighbors")
#dev.off()
testData <- test
#####
#Extract row- and column-wise feature data
#rm(features)
setwd("~/Documents/_Documents/Reno/Reno_2019-Spring/MachineLearning/Homework1/Features")
load("Features.R")

#Initialize storage matrix for test point predictions (Set to '-1')
tmpTestPredictions <- rep(-1, dim(testData)[3])

#Iterate across all test points
for(t in 165:dim(testData)[3])
{
  print(t)
  #Initialize temporary density storage info
  tmpDensity <- rep(NA, 16)
  
  #Round data to one digit
  currentObs <- round(testData[,,t], 1)
  
  #Extract row-wise and column-wise sum densities
  tmpRowSumDensity <- density(rowSums(currentObs))$y
  tmpColSumDensity <- density(colSums(currentObs))$y
  
  #Extract summary statistics, std dev, and number of modes from density of row-wise sums
  tmpDensity[1] <- min(tmpRowSumDensity)                 #Min
  tmpDensity[2] <- quantile(tmpRowSumDensity, .25)       #1st quartile
  tmpDensity[3] <- quantile(tmpRowSumDensity, .50)       #Median
  tmpDensity[4] <- mean(tmpRowSumDensity)                #Mean
  tmpDensity[5] <- quantile(tmpRowSumDensity, .75)       #3rd quartile
  tmpDensity[6] <- max(tmpRowSumDensity)                 #Max
  tmpDensity[7] <- sd(tmpRowSumDensity)                  #Std dev
  tmpDensity[8] <- length(findModes(tmpRowSumDensity))  #Number of modes
  
  #Extract summary statistics, std dev, and number of modes from density of column-wise sums
  tmpDensity[9] <- min(tmpColSumDensity)                 #Min
  tmpDensity[10] <- quantile(tmpColSumDensity, .25)       #1st quartile
  tmpDensity[11] <- quantile(tmpColSumDensity, .50)       #Median
  tmpDensity[12] <- mean(tmpColSumDensity)                #Mean
  tmpDensity[13] <- quantile(tmpColSumDensity, .75)       #3rd quartile
  tmpDensity[14] <- max(tmpColSumDensity)                 #Max
  tmpDensity[15] <- sd(tmpColSumDensity)                  #Std dev
  tmpDensity[16] <- length(findModes(tmpColSumDensity))  #Number of modes
  
  
  #Calculate distance between test point info and EACH other point
  testDistances <- calculateDistances(tmpDensity)
  
  
  if(t == 1)
  {
    allDistances <- testDistances
  } else
  {
    allDistances <- cbind(allDistances,testDistances[,2])
  }
}


dim(allDistances)
head(allDistances)
