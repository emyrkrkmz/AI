#	*Variable Types*
#
#	1)Numerical variables
#
#	2)Categorical variables
#		a)Nominal
#		b)Ordinal
#	
#	3) Dependent Variables -> Target, dependent, output, response
#	
#	4) Independent Variables -> Feature, independent, input, column, predictor, explanatory

#	*Learning Types*
#
#	1) Supervised Learning
#
#	2) Unsuprevised Learning
#
#	3) Reinforcement Learning
#

#	*Problem Types*
#
#	1) Regression Problems
#
#	2) Classification Problems


#	*Model success evaluation*
#
#	For numerical problems
#
#	MSE = 1/n (summation i=1 to n)[∑(yi - (y^)i)**2]	yi = Real value, y^= Predicted value
#	RMSE = √MSE
#	MAE = 1/n (summation i=1 to n)[∑|yi - (y^)i|]
# 
#	For classification problems -> Accuracy = # of correct classification / # of total classification
#

#	*Model Validation*
#
#	1) Holdout Validation -> Parse Original dataset to Education dataset and Test dataset
#	!How should we parse!
#	
#	2) K Fold Cross Validation -> Parse Original dataset to 5 part. Make 4 part Education dataset and Test on 1 part OR
#								  Parse Original dataset like Holdout val. and make Cross val on just Education dataset,
#								  after that use on test set.
#

#	*Bias-Variance Tradeoff*
#
#	Underfitting	-	Correct Model	-	 Overfitting
#	  High bias		Low bias, low variance	High Variance
#	
# 	What is overfitting?
#