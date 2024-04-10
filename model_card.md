# Model Card


## Model Details

The deployed model is a Random Forest Classifier trained on Census data. It is used to predict whether an individual earns more or less than $50,000 annually based on numerous variables.

## Intended Use

The model is intended to be used for predicting income levels based on demographic and employment information. 

## Training Data

The model was trained using information from the Census data. This data includes details like age, education, job, and marital status, along with income levels labeled as '>50K' and '<=50K'. 80% of the Census data is used for training data.

## Evaluation Data

To test the model's performance, a separate set of data was kept aside from the same Census data. This helps to check how well the model works on new, unseen data without any bias from the training process. 20% of the Census data is used for evaluation data.

## Metrics

Precision: 0.7427
Recall: 0.6340
F1 Score: 0.6841

## Ethical Considerations

The model should be evaluated for fairness across different demographic groups to ensure it does not exhibit bias against any particular group. It is also essential to handle sensitive information such as demographic data carefully and ensure compliance with privacy regulations.

## Caveats and Recommendations

Be aware of potential biases in the training data which would lead to biased predictions. The model may need to be updated with new data periodically to ensure there is no data bias.
