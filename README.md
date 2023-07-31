# kd-tree-and-forest

Wine experts evaluate the quality of wine based on sensory data. We could also collect the features of wine from objective tests, thus the objective features could be used to predict the expert’s judgement, which is the quality rating of the wine. This could be formed as a supervised learning problem with the objective features as the data features and wine quality rating as the data labels.

In this project, the data provides objective features obtained from physicochemical statistics for each white wine sample and its corresponding rating provided by wine experts. 

k-d tree (KDT), and random forest were inplemented to provide wine quality prediction on the test set.

Wine quality rating is measured in the range of 0-9. In our dataset, we only keep the samples for quality ratings 5, 6 and 7. The 11 objective features are listed as follows:

• f acid: fixed acidity
• v acid: volatile acidity
• c acid: citric acid
• res sugar: residual sugar
• chlorides: chlorides
• fs dioxide: free sulfur dioxide 
• ts dioxide: total sulfur dioxide 
• density: density
• pH: pH
• sulphates: sulphates 
• alcohol: alcohol

__main algorithms can be found in `main_algorithm` file.__
