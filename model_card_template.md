# Model Card

Model card includes information about salary_predictor model.

## Model Details
Model is a logistic regression model developed using scikit-leran library.

## Intended Use
Model can be used for predicting salary information.

## Training Data
Publicly available census.csv data is used for training and evaluation. 80% of the data is used for training purposes.

## Evaluation Data
20% of the census.csv data held out for evaluation.

## Metrics
For monitoring the performance of the model f_beta, precision and recall have been used. 

Overall performance of the model on test set as follows;
precision: 0.7285, recall:0.2699, f_beta: 0.3939

## Ethical Considerations
Slice results revealed that the model doesn't reflect the same performances in all situations. For example, for females the precision significantly decrease to 0.5383. This shows the model has a bias.

## Caveats and Recommendations
In order to improve the performances the model could be trained with more balanced data. Also using more complext model architecture could yiled better performance.