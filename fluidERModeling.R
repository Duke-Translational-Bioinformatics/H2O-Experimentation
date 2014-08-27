## fluidERModeling.R
## erich.huang@duke.edu

## Experimenting with the H2O framework for machine learning

## REQUIRED PACKAGES
require(h2o)

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

## DIVIDE DATA INTO TRAIN AND TEST SETS
set.seed(140827)
trainIDs <- sample(rownames(vdxExpDF), size = 0.66*nrow(vdxExpDF))
testIDs <- setdiff(rownames(vdxExpDF), trainIDs)

trainData <- vdxExpDF[trainIDs, ]
testData <- vdxExpDF[testIDs, ]

## PUSH DATA to H2O
trainBC <- as.h2o(client = localH2O, object = trainData, key = 'trainBC')
testBC <- as.h2o(client = localH2O, object = testData, key = 'testBC')

## BUILD MODEL
rfModel <- h2o.randomForest(x = 3:ncol(testBC), 
                            y = 2,
                            data = testBC, 
                            classification = TRUE, 
                            ntree = 500, 
                            importance = TRUE, 
                            balance.classes = TRUE,
                            version = 2)

