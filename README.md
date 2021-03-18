# Machine-Learning-Metrics-Package-Tensorflow-PyTorch-Keras
An significant aspect of a project is testing the machine learning algorithm. If you evaluate using an index, the model could offer satisfactory results. But, if you evaluate against other indicators such as logarithmic loss or some other such measure, you may have bad results. Most commonly we use classification precision to calculate our model's efficiency, but it is not adequate to really assess our model. In this repo, various forms of metrics for different types of models/applications are covered.

 * **Classification**
   1. Classification Accuracy
   4. Confusion matrix
   5. Precision and Recall
   6. F-measure
   7. Receiver Operating Characteristic (ROC)
   8. Area Under Curve (AUC)
   9. Precision Recall Curve
 * **Segmentation** 
   1. Intersection over Union (IOU)
   2. Dice coefficient
   3. Pixel Accuracy
   4. Precision and Recall
   5. Confusion matrix
   6. Receiver Operating Characteristic (ROC)
   7. Area Under Curve (AUC)
 * **Object Detection** 
   1. Average Precision (Pascal)
   2. Average Precision (COCO)
   3. Confusion Matrix
   4. PR Curve
## Necessary Imports
```python
# for creating custom metrics
import numpy as np
# for plotting
import matplotlib.pyplot as plt
# for using scikit-learn's built-in metrics
from sklearn.metrics import *
# for using tesnorflow/keras' built-in metrics
import tensorflow as tf
import tensorflow.keras.backend as K
```
## Note
When you training your model in tensorflow or keras and you use your custom/built-in metrics for evaluation of you model then the results deiplayed after every iteration are only for the batch the network iterated over not the average score over all data. For example, at the end of evaluation when you run the following command
```python
model.evaluate(test_data_gen)
```
you see output;
```
355/354 [==============================] - 504s 1s/step - loss: 0.0927 - accuracy: 0.867 
```
but in this case the value 86.7% is not the accuracy over all the test data it is the accuracy over the last batch that passed through the network. So, to properly evaluate your model save your prdictions and gorund truth in numpy arrays or tensors, like [y_true, y_pred], (in appropriate format depending upon the network type) and then run the custom or built-in metrics over all the data to get the average value of your metric.

## Classification

### Classification Accuracy

img1.png

Works well if there are equal number of samples belonging to each class, 
For example:
Class A : 98% samples 
Class B : 2% samples of class B in our training set. 
Then our model can easily get 98% training accuracy by simply predicting every training sample belonging to class A.
Classification Accuracy is great, but gives us the false sense of achieving high accuracy.
The real problem arises, when the cost of misclassification of the minor class samples are very high. 
For example: If we deal with a rare but fatal disease, the cost of failing to diagnose the disease of a sick person is much higher than the cost of sending a healthy person to more tests.
```python
# for using scikit-learn
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
# for tf/keras if you are using built-in metrics
# define in model.compile like
model.compile(loss=LOSS_FUNCTION, optimizer=Adam(), metrics='accuracy')
```
if you are using `tensorflow` version >= 2.0.0 then you can also use Top K accuracy, usually used in `ImageNet` classification paper, like;
```python
# define
Top_K = tf.keras.metrics.TopKCategoricalAccuracy(k=top_k)
# then
model.compile(loss=LOSS_FUNCTION, optimizer=Adam(), metrics=['accuracy', Top_K])
```

### Confusion Matrix
A table used to describe the performance of a classification model on a test data
Allows easy identification of confusion between classes
img5
Class 1 : Positive
Class 2 : Negative
Positive (P) : Observation is positive (for example: is an apple).
Negative (N) : Observation is not positive (for example: is not an apple).
True Positive (TP) : Observation is positive, and is predicted to be positive.
False Negative (FN) : Observation is positive, but is predicted negative.
True Negative (TN) : Observation is negative, and is predicted to be negative.
False Positive (FP) : Observation is negative, but is predicted positive.
img6

img7
Assumes equal cost for both kinds of errors
Does not perform well with imbalanced data 
For this you can define a `callback` in tensorflow so that the confusion matrix is logged in you plotting pane or in the tensorboard after every epoch
```python
# tensorflow/keras
def plot_confusion_matrix(cm, class_names, normalize=True):
    
    conf_mat = cm
    conf_mat = cm
    if normalize:
        row_sums = conf_mat.sum(axis=1)
        conf_mat = conf_mat / row_sums[:, np.newaxis]
        conf_mat = np.round(conf_mat, 3)
    my_cmap = 'CMRmap'# viridis, seismic, gray, ocean, CMRmap, RdYlBu, rainbow, jet, Blues, Greens, Purples
    
    x_labels = class_names
    y_labels = class_names
    c_m = conf_mat
    
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(c_m, cmap = my_cmap) 
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")#ha=right
    
    # Loop over data dimensions and create text annotations.
    def clr_select(i, j):
        if i==j:
            color="green"
        else:
            color="red"
        return color
    
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, c_m[i, j], color="k", ha="center", va="center")#color=clr_select(i, j)
    
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm)
    plt.show() 
    return fig     


class log_confusion_matrix(Callback):
        '''
        Decalre Arguments Input Here
        '''
        def __init__(self, test_gen, class_names, log_dir):
            self.test_gen = test_gen
            # self.test_images = self.test_gen.next()[0]
            # self.test_labels = self.test_gen.next()[1]
            self.class_names = class_names
            self.log_dir = log_dir
            
        #def on_epoch_end(self, epoch, logs={}):
        def on_train_end(self, epoch, logs={}):
            total_test_pred = []
            total_test_labels_bin = []
            for i in range(len(self.test_gen)):
                test_images, test_labels = self.test_gen.next()
                # Use the model to predict the values from the validation dataset.
                test_pred_raw = self.model.predict(test_images)
                
                test_pred = np.argmax(test_pred_raw, axis=1)
                test_labels_bin = np.argmax(test_labels, axis=1)
                
                total_test_pred.append(test_pred)
                total_test_labels_bin.append(test_labels_bin)
                
        
            total_test_pred = np.concatenate(total_test_pred, axis=0)
            total_test_labels_bin = np.concatenate(total_test_labels_bin, axis=0)
            
            total_test_pred = np.resize(total_test_pred, (-1, 1)).astype(np.int64)
            total_test_labels_bin = np.resize(total_test_labels_bin, (-1, 1)).astype(np.int64)
            # Calculate the confusion matrix.
            cm = confusion_matrix(total_test_labels_bin, total_test_pred, labels=list(np.arange(0,len(self.class_names))))
            # Log the confusion matrix as an image summary.
            figure = plot_confusion_matrix(cm, class_names=self.class_names)
            figure.savefig('../data/cm.jpg', format='jpg', dpi=400)
```
```python
# for sk-learn
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
```
### Precision and Recall

Precision: Total number of correctly classified positive examples divided by the total number of predicted positive examples
img2
Recall: Total number of correctly classified positive examples divided by the total number of positive examples
img3
* High recall, high precision 
Desired system 
* High recall, low precision
This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives.
* Low recall, high precision
This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP)

```python
# tensorflow/keras
def recall_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

```
```python
# sk-learn
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())# ravel is just flattening the arrays
```
### F-Measure
A measure that combines precision and recall
Harmonic mean of precision and recall 
Punishes the extreme values more
Will always be nearer to the smaller value of Precision or Recall
Reaches its best value at 1 and worst at 0

img4

A more general F score, that uses a positive real factor β, where β is chosen such that recall is considered β times as important as precision, is:

img8
```python
# tensorflow /keras
def F_Measure(y_true, y_pred):
    
    y_true = tf.cast(y_true, "float32")    
    y_pred = tf.cast(y_pred, "float32")
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
```
```python
# sk-learn
from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
f1_score(y_true, y_pred, average='macro')
```
### ROC Curve
Graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied.
Plot is created by plotting the true positive rate (recall, sensitivity) against the false positive rate (FPR) at various thresholds. 

img9
img10
img11
img12
img13

curnves are usually drawn after full training. 
(A more detailed version of this in Segmentation Metrics section in same repo)*
```python
# a more detailed version of this in Segmentation Metrics section in same repo
from sklearn.metrics import roc_curve, auc

img_roc = np.load('../data/Img_roc.npy')/255 # noramalized images a tensor or 4D Array [batch, H, W, 3]
gt_roc = np.load('../data/GT_roc.npy') # GT saved in numpy array in same format as the network's output; e.g. for classification 2D array [Batch, num_classes]

preds_val = model.predict(img_roc, verbose=1)

plt.figure(figsize=(7, 7))
for i in range(num_class):
    clr = ['r', 'g', 'c', 'm']
    l_style = [':','--', '-.', '-' ]
    
    ground_truth_labels = gt_roc[:,:,:,i].ravel() 
    score_value = preds_val[:,:,:,i].ravel() 
    
    fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr)
   
    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, color=clr[i], linestyle=l_style[i], label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');
```
### Aread Under Curve
One of the most widely used metrics for evaluation of binary classifier
Aggregate measure of performance across all classification thresholds
“AUC of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example.” 

img14

AUC represents the probability that a random positive example is positioned to the right of a random negative example.
AUC ranges from 0 to 1.
AUC is classification-threshold-invariant

img15
Look at ROC curve. (A more detailed version of this in Segmentation Metrics section in same repo)*
### Precision Recall Curve
A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds, much like the ROC curve.
Precision-Recall curves should be used when there is a moderate to large class imbalance.

img16
(A more detailed version of this in Segmentation Metrics section in same repo)*
```python
# a more detailed version of this in Segmentation Metrics section in same repo
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

img_roc = np.load('../data/Img_roc.npy')/255 # noramalized images a tensor or 4D Array [batch, H, W, 3]
gt_roc = np.load('../data/GT_roc.npy') # GT saved in numpy array in same format as the network's output; e.g. for classification 2D array [Batch, num_classes]

preds_val = model.predict(img_roc, verbose=1)

plt.figure(figsize=(7, 7))
for j in range(num_class):
    clr = ['r', 'g', 'c', 'm']
    l_style = [':','--', '-.', '-' ]
    
    ground_truth_labels = gt_roc[:,:,:,j].ravel() 
    score_value = preds_val[:,:,:,j].ravel()
    
    precision, recall, thresholds = precision_recall_curve(ground_truth_labels, score_value)
    mAP = average_precision_score(ground_truth_labels, score_value)
   
    plt.title("Precision Recall Curve")
    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(recall, precision, color=clr[j], linestyle=l_style[j], label = 'mAP (area = %0.2f)' % mAP)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='lower left');
```
