# Model Card

## Model Details
This model was created by [@culv](https://github.com/culv). It is a Random Forest Classifier model trained using the default hyperparameters in scikit-learn version 1.3.1

## Intended Use
This model should be used to predict whether an individual's salary is above or below $50,000 per year based on their census data

## Training Data
This model was trained on the [Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income) from the UCI Machine Learning Repository. The data was pulled from the 1994 US census. Categorical features were encoded using a one-hot encoder, and target labels were encoded using a label binarizer.

## Evaluation Data
This model was evaluated on a holdout test set consisting of 20% of the total data. In addition to computing metrics on the overall dataset, metrics were also computed over slices of each categorical feature (e.g. sex=Male and sex=Female) 

## Metrics
This model has a precision of 0.623, a recall of 0.717, and a fbeta score of 0.667 on the entire test set. Metrics for slices of categorical features can also be found [here](ml/slice_performance.txt)

## Ethical Considerations
This model contains bias for certain slices. For example, the model has higher fbeta scores for countries with more representation in the dataset like the US, and lower fbeta scores for countries with lower representation like Canada or Cuba. Similarly, the model has better performance on individuals whos profession is "Tech-support" compared to individuals whos profession is "Handler-cleaner" 

## Caveats and Recommendations
When using predictions from this model, we must remember that the model was trained on an imbalanced dataset that had a higher representation of certain demographics