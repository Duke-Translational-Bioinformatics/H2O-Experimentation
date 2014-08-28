## fluidERModeling.R
## erich.huang@duke.edu

## Experimenting with the H2O framework for machine learning

## REQUIRED PACKAGES
require(h2o)
require(ggplot2)

## INITIALIZE H2O
localH2O <- h2o.init(ip = 'localhost',
                     port = 54321,
                     startH2O = TRUE,
                     Xmx = '4g')

## LOAD DATA
# loads breast cancer data aggregatged by Quackenbush's Harvard research group
# fullDF = expression data frame with patients as rows and probesets as columns
# metaDF = clinical data frame with patients as rows and clinical covariates as columns
load('~/Quantwork/ARTEMIS/bcSignaling/data/dfciData.RData')

# For the sake of experimenting with the H2O framework, let's subset the data some.
metaDF$series <- as.factor(metaDF$series)
vdxMetaDF <- metaDF[metaDF$series == 'VDX', ]
vdxExpDF <- fullDF[grep('VDX_', rownames(fullDF)), ]

# Reduce the feature space for testing
set.seed(140826)
sampleGenes <- sample(colnames(vdxExpDF), size = 500)
vdxExpDF <- vdxExpDF[ , sampleGenes]

vdxExpDF <- data.frame(patients = rownames(vdxExpDF), er_status = vdxMetaDF[rownames(vdxExpDF), 'er'], vdxExpDF)

# Find NA for ER status
naInd <- grep('TRUE', is.na(vdxExpDF$er_status))

vdxExpDF <- vdxExpDF[-naInd, ]

## DIVIDE DATA INTO TRAIN AND TEST SETS
set.seed(140827)
trainIDs <- sample(rownames(vdxExpDF), size = 0.66*nrow(vdxExpDF))
testIDs <- setdiff(rownames(vdxExpDF), trainIDs)

trainData <- vdxExpDF[trainIDs, ]
testData <- vdxExpDF[testIDs, ]

## PUSH DATA to H2O
trainBC <- as.h2o(client = localH2O, object = trainData, key = 'trainBC')
testBC <- as.h2o(client = localH2O, object = testData, key = 'testBC')

## BUILD A DISTRIBUTED RANDOM FOREST MODEL
rfModel <- h2o.randomForest(x = 3:ncol(trainBC), 
                            y = 2,
                            data = trainBC, 
                            classification = TRUE, 
                            ntree = 500, 
                            importance = TRUE, 
                            balance.classes = TRUE,
                            version = 2)

## PREDICT ON TEST SET
yHatTest <- h2o.predict(rfModel, testBC)
yHatTestDF <- as.data.frame(yHatTest)
colnames(yHatTestDF) <- c('ER_Call', 'ER_Neg', 'ER_Prob')
yHatTestDF <- data.frame('ER_True' = testData[ , 'er_status'], yHatTestDF)

ggplot(yHatTestDF, aes(factor(ER_True), ER_Prob)) +
  geom_boxplot(aes(fill = factor(ER_True)), alpha = 0.4) +
  geom_jitter(aes(colour = factor(ER_True)), size = 4) +
  ylim(c(0.0, 1.0)) +
  ggtitle('Distributed Random Forest ER Predictions on Held-Out Test Data')

rfPerformance <- h2o.performance(data = yHatTest[ , 3], reference = testBC[ , 2])

## BUILD A DEEP LEARNING MODEL
# dlModel <- h2o.deeplearning(x = 3:ncol(trainBC),
#                             y = 2,
#                             data = trainBC,
#                             activation = 'TanhWithDropout',
#                             input_dropout_ratio = 0,
#                             hidden_dropout_ratios = c(0.7, 0.7, 0.7),
#                             hidden = c(50, 50, 50),
#                             epochs = 500,
#                             classification = TRUE,
#                             balance_classes = TRUE)

dlModel <- h2o.deeplearning(x = 3:ncol(trainBC),
                            y = 2,
                            data = trainBC,
                            classification = TRUE,
                            balance_classes = TRUE)
  
## PREDICT ON TEST SET
dlYHatTest <- h2o.predict(dlModel, testBC)
dlYHatTestDF <- as.data.frame(dlYHatTest)  
colnames(dlYHatTestDF) <- c('ER_Call', 'ER_Neg', 'ER_Prob')
dlYHatTestDF <- data.frame('ER_True' = testData[ , 'er_status'], dlYHatTestDF)

ggplot(dlYHatTestDF, aes(factor(ER_True), ER_Prob)) +
  geom_boxplot(aes(fill = factor(ER_True)), alpha = 0.4) +
  geom_jitter(aes(colour = factor(ER_True)), size = 4) +
  ylim(c(0.0, 1.0)) +
  ggtitle('Regularized Deep Learning ER Predictions on Held-Out Test Data')

dlPerformance <- h2o.performance(data = dlYHatTest[ , 3], reference = testBC[ , 2])


