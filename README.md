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
**Precision** effectively describes the purity of our positive detections relative to the ground truth. How many of these things were actually annotated in the ground truth of all the items we predicted in a given picture.
Precision: Total number of correctly classified positive examples divided by the total number of predicted positive examples

img2
**Recall** describes the completeness of our positive predictions relative to the ground truth. Of all of the objected annotated in our GT, how many did the network detected as positive predictions.
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

## Semantic Segmentation

### Intersection over Union (IOU) (Jaccard Index)
The Intersection over Union (IoU) metric, also referred to as the Jaccard index, is essentially a method to quantify the percent overlap between the target mask and our prediction output. This metric is closely related to the Dice coefficient which is often used as a loss function during training.

img17

The intersection (A∩B) is comprised of the pixels found in both the prediction mask and the ground truth mask, whereas the union (A∪B) is simply comprised of all pixels found in either the prediction or target mask.

img 18

For multi-classes calculate the IoU score for each class separately, average over all classes to provide a global, mean IoU score of semantic segmentation

img19

```python
def mean_iou(y_true, y_pred, smooth=1):
    
    #y_true = y_true * 255 # if tf.data_gen has rescaled the iamges
    
    
    if y_pred.shape[-1] <= 1:# for binary segmentation
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:# for multi-class segmentation
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou
```
As I already said in the **Note** its better to evalurte your model on the whole dataset after training, so use the following `numpy` function to calculate the iou. and then you can average the mean IOU, of one image for all classes, over all the dataset/images
```python
def Class_Wise_IOU(Pred, GT, NumClasses, ClassNames, display=False):
    '''
    Parameters
    ----------
    Pred: 2D array containing unique values for N classes [0, 1, 2, ...N]
    GT: 2D array containing unique values for N classes [0, 1, 2, ...N]
    NumClasses: int total number of classes including BG
    ClassNames : list of classes names
    Display: if want to print results
    Returns
    -------
    mean_IOU: mean over classes that are present in the GT (other classes are ignored)
    ClassIOU[:-1] : IOU of all classes in order
    ClassWeight[:-1] : no. of pixles in Union of each class present

    '''
    #Given A ground true and predicted labels per pixel return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate class intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate class Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
            
    # b/c we will only take the mean over classes that are actually present in the GT
    present_classes = np.unique(GT) 
    mean_IOU = np.mean(ClassIOU[present_classes])
    # append it in final results
    ClassNames = np.append(ClassNames, 'Mean')
    ClassIOU = np.append(ClassIOU, mean_IOU)
    ClassWeight = np.append(ClassWeight, np.sum(ClassWeight))
    if display:
        result = np.concatenate((np.asarray(ClassNames).reshape(-1,1), 
                                 np.round(np.asarray(ClassIOU).reshape(-1,1),4),
                                 np.asarray(ClassWeight).reshape(-1,1)), 1)
        print(tabulate(np.ndarray.tolist(result), headers = ["Classes","IoU", "Class Weight(# Pixel)"], tablefmt="github"))
    
    return mean_IOU, ClassIOU[:-1], ClassWeight[:-1]
```
#### Note: A function purely implemented purely in numpy can be converted into the `tensorflow` function and can be used inside the `model.compile` function to get output after every iteration. 
Following is an example code to do just that.
```python
def Strict_IOU(Pred, GT, NumClasses, ClassNames):
    '''
    Parameters
    ----------
    Pred: 2D array containing unique values for N classes [0, 1, 2, ...N]
    GT: 2D array containing unique values for N classes [0, 1, 2, ...N]
    NumClasses: int total number of classes including BG
    ClassNames : list of classes names
    Display: if want to print results
    Returns
    -------
    mean_IOU: mean over classes that are present in the GT (other classes are ignored)
    ClassIOU[:-1] : IOU of all classes in order
    ClassWeight[:-1] : no. of pixles in Union of each class present

    '''
    #Given A ground true and predicted labels per pixel return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate class intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate class Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
            
    # b/c we will only take the mean over classes that are actually present in the GT
    present_classes = np.unique(GT) 
    mean_IOU = np.mean(ClassIOU[present_classes])
    # append it in final results
    ClassNames = np.append(ClassNames, 'Mean')
    ClassIOU = np.append(ClassIOU, mean_IOU)
    ClassWeight = np.append(ClassWeight, np.sum(ClassWeight))
    
    return mean_IOU

NumClasses=6
ClassNames=['Background', 'Class_1', 'Class_1',
            'Class_1 ', 'Class_1', 'Class_1 ']

def strict_iou(y_true, y_pred):
    
    '''
    only supported for btach size 1
    '''
    y_true = K.squeeze(y_true, 3)#[? H W 1] -> [? H W]
    y_true = K.squeeze(y_true, 0)#[H W] -> [H W]
    y_true = tf.cast(y_true, "int32")#[H W] -> [H W]
    
    
    y_pred = tf.keras.activations.softmax(y_pred, axis=-1)#[? H W Ch] -> [? H W Ch]
    y_pred = tf.cast(y_pred > 0.5, "int32")#[? H W Ch] -> [? H W Ch]
    y_pred = tf.math.argmax(y_pred, axis=-1)#[? H W CH] -> [? H W]
    y_pred = K.squeeze(y_pred, 0)#[? H W] -> [H W]
    
    x = tf.numpy_function(Strict_IOU, [y_pred, y_true, NumClasses, ClassNames], 
                          tf.float64, name=None)
    return x
```
then use like
```python
model.compile(optimizer=Nadam(), loss=LOSS, metrics=strict_iou)
```
## Dice Coefficient
Dice coefficient, which is essentially a measure of overlap between two samples. This measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap. The Dice coefficient was originally developed for binary data, and can be calculated as:

img20

where |A∩B| represents the common elements between sets A and B, and |A| represents the number of elements in set A (and likewise for set B).
For details: (here)[https://www.jeremyjordan.me/semantic-segmentation/#loss]

```python
def dice_coef(y_true, y_pred, smooth=2):
    
    #y_true = y_true * 255 # if tf.data_gen has rescaled the iamges
    
    if y_pred.shape[-1] <= 1:# for binary segmentation
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:# for multi-class segmentation
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=[0])
    return dice
```
## Pixel Accuracy
One alternative method of evaluating semantic segmentation is only to report the percentage of correctly labeled pixels in the image. The precision of the pixel is often recorded both individually and globally for each class.
img21
This metric is also inaccurate if the class representation is small in the picture, as the measure will be biased towards reporting how well you identify negative case (i.e., where the class is not present).

## Precision, Recall and F1-Measure
Same as explaind the **Classifiction** section. We do need to modify the function a little bit.
```python
def recall_m(y_true, y_pred):

    #y_true = y_true * 255 # if tf.data_gen has rescaled the iamges
    
    if y_pred.shape[-1] <= 1:# for binary segmentation
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:# for multi-class segmentation
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    return recall

def precision_m(y_true, y_pred):
    
    #y_true = y_true * 255 # if tf.data_gen has rescaled the iamges
    
    if y_pred.shape[-1] <= 1:# for binary segmentation
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:# for multi-class segmentation
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    return precision

def F_Measure(precision, recall):
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
```
## Receiver Operator Characteristic (ROC) Curve
The explaination is similar to as explained in **Classification** section.
The modified code for segmentation model is as follows 
**___Detailed Code__**

```python
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||Data Process|||||||||||||||||||||||||||||||||||||||||||||||||
'''
    1. It takes inputs of the same size as the 'model'.
    2. Make the unnormalized tensors of the images(BxWxHx3) and their corresponding groundtruths(BxWxH).
    3. It'll normalize and convert them to 1hot itself and process them futher for ROC and PR curve.
    4.Line style and color ranges are defined for maximum 7 classes. If model has more classes then 
      just remove the 'clr' and 'l_style' lists.
    5. All the functions take different types of inputs shapes so be carefull wehn assigning them.
'''
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, average_precision_score, roc_curve, auc
from itertools import cycle


img_roc = np.load('../data_ssd/Talha/images.npy')/255
gt_roc = np.load('../data_ssd/Talha/masks.npy',0)# A 3D array of shape [total_images, H, W], each [H, W] mask will contain values from [0, 1, 2, ..., C] where C is total number                                                  # of classes in dataset
# gt_roc = gt_roc * (num_class + 1)# case specific rescaling
gt_roc = tf.cast(gt_roc, "int32")
gt_roc = tf.one_hot(gt_roc, num_class, axis=-1)
sess = tf.compat.v1.Session()
gt_roc = sess.run(gt_roc)

preds_val = model.predict(img_roc, batch_size=2, verbose=1)

gt_pr = gt_roc.reshape(-1,3)
preds_pr = preds_val.reshape(-1,3)
_,n_classes = gt_pr.shape
```
This above script will provide the inputs for the next three scripts (`ROC curve`, `PRcurve` and `Confusion_matrix`).
```python

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||ROC Curves|||||||||||||||||||||||||||||||||||||||||||||||||
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(gt_pr[:, i], preds_pr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(gt_pr.ravel(), preds_pr.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw = 2 #line width
# Plot all ROC curves
plt.figure(figsize=(7, 7))
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
l_style = cycle([':','--', '-.', '-'])
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
class_name = cycle(["Ripe", "Unripe", "Green"])
for i, name, color, line_style in zip(range(n_classes), class_name, colors,  l_style):
    plt.plot(fpr[i], tpr[i], color=color, linestyle=line_style,lw=lw,label='ROC curve for {0} Strawberry(area = {1:0.2f})'''.format(name, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```
## Precision-Recall (PR) Curve
Followed by `Data Process` Script (above) Continued*
```python
#|||||||||||||||||||||||||||||||||||||||||||||||||Plotting Percision Recall Curves|||||||||||||||||||||||||||||||||||||||||||||||||
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(gt_pr[:, i], preds_pr[:, i])
    average_precision[i] = average_precision_score(gt_pr[:, i], preds_pr[:, i])
# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(gt_pr.ravel(),preds_pr.ravel())
average_precision["micro"] = average_precision_score(gt_pr, preds_pr,average="micro")
average_precision["macro"] = average_precision_score(gt_pr, preds_pr,average="macro")
#print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'aqua'])
l_style = cycle([':','--', '-.', '-'])
plt.figure(figsize=(7, 9))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')

l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

l, = plt.plot(recall["micro"], precision["micro"], color='red',linestyle=':', lw=2)
lines.append(l)

labels.append('macro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["macro"]))
class_name = cycle(["Ripe", "Unripe", "Green"])
for i, name, color, line_style in zip(range(n_classes), class_name, colors, l_style):
    l, = plt.plot(recall[i], precision[i], color=color, linestyle=line_style, lw=2)
    lines.append(l)
    labels.append('Precision-recall for {0} Strawberry(area = {1:0.2f})'''.format(name, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
#plt.plot([0, 1],[1, 0], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
#plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.legend(lines, labels,loc="lower left")
plt.show()
```
## Confusion matrix
Followed by `Data Process` Script (above) Continued*
```python
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||Plotting  COnfusion matrix|||||||||||||||||||||||||||||||||||||||||||||||||||||||
from sklearn.metrics import confusion_matrix


gt_conf = (np.argmax(gt_roc, 3).reshape(-1,1))
preds_conf = (np.argmax(preds_val, 3).reshape(-1,1))

y_true = gt_conf
y_pred = preds_conf

conf_mat = np.round(confusion_matrix(y_true, y_pred, normalize='true'), 3)

include_BG = True
my_cmap = 'Greens'# viridis, seismic, gray, ocean, CMRmap, RdYlBu, rainbow, jet, Blues, Greens, Purples
if include_BG == False:
    x_labels = ["Ripe", "Unripe", "Green"]
    y_labels = ["Ripe", "Unripe", "Green"]
    c_m = conf_mat[1:4, 1:4]
else:
    x_labels = ["BG", "Ripe", "Unripe", "Green"]
    y_labels = ["BG", "Ripe", "Unripe", "Green"]
    c_m = conf_mat
   
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(c_m, cmap = my_cmap) 

# We want to show all ticks...
ax.set_xticks(np.arange(len(y_labels)))
ax.set_yticks(np.arange(len(x_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(y_labels)
ax.set_yticklabels(x_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")#ha=right

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
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
plt.colorbar(sm)
plt.show() 
```
## Object Detection

### Average Precision (PASCAL & COCO)
A very good explaination for these metrics is given by [rafaelpadilla here](https://github.com/rafaelpadilla/Object-Detection-Metrics).
I don't think I can explain it better than him so have a look at his repo you can also use his repo for evaluating your own models.

### Confusion Matrix for Object Detection 
After you have evaluated your object detection model either via Pascal VOC or COCO average precision, now you can use those results to built a confusion matrix have a look at my [repo here](https://github.com/Mr-TalhaIlyas/Confusion_Matrix_for_Objecti_Detection_Models).
