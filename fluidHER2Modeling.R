## fluidHER2Modeling.R
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

# Subset the data on samples annotated for HER2 status
her2Ind <- grep('FALSE', is.na(metaDF$her2))
her2SampIDs <- metaDF$samplename[her2Ind]

vdxMetaDF <- metaDF[her2SampIDs, ]
vdxExpDF <- fullDF[her2SampIDs, ]

# Reduce the feature space for testing
set.seed(140826)
sampleGenes <- sample(colnames(vdxExpDF), size = 500)
vdxExpDF <- vdxExpDF[ , sampleGenes]

vdxExpDF <- data.frame(patients = rownames(vdxExpDF), her2_status = vdxMetaDF[rownames(vdxExpDF), 'her2'], vdxExpDF)

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
colnames(yHatTestDF) <- c('HER2_Call', 'HER2_Neg', 'HER2_Prob')
yHatTestDF <- data.frame('HER2_True' = testData[ , 'her2_status'], yHatTestDF)

ggplot(yHatTestDF, aes(factor(HER2_True), HER2_Prob)) +
  geom_boxplot(aes(fill = factor(HER2_True)), alpha = 0.4) +
  geom_jitter(aes(colour = factor(HER2_True)), size = 4) +
  ggtitle('Distributed Random Forest HER2 Prediction on Held-Out Test Set\n')

rfPerformance <- h2o.performance(data = yHatTest[ , 3], reference = testBC[ , 2])

## BUILD A DEEP LEARNING MODEL
dlModel <- h2o.deeplearning(x = 3:ncol(trainBC),
                            y = 2,
                            data = trainBC,
                            activation = 'TanhWithDropout',
                            input_dropout_ratio = 0,
                            hidden_dropout_ratios = c(0.5, 0.5, 0.5),
                            hidden = c(50, 50, 50),
                            epochs = 500,
                            classification = TRUE,
                            balance_classes = TRUE)
  
## PREDICT ON TEST SET
dlYHatTest <- h2o.predict(dlModel, testBC)
dlYHatTestDF <- as.data.frame(dlYHatTest)  
colnames(dlYHatTestDF) <- c('HER2_Status', 'HER2_Neg', 'HER2_Prob')
dlYHatTestDF <- data.frame('HER2_True' = testData[ , 'her2_status'], dlYHatTestDF)

ggplot(dlYHatTestDF, aes(factor(HER2_Status), HER2_Prob)) +
  geom_boxplot(aes(fill = factor(HER2_Status)), alpha = 0.4) +
  geom_jitter(aes(colour = factor(HER2_Status)), size = 4) +
  ggtitle('Regularized Deep Learning HER2 Prediction on Held-Out Test Set\n')

dlPerformance <- h2o.performance(data = dlYHatTest[ , 3], reference = testBC[ , 2])


