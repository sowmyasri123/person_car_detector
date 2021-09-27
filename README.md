# person_car_detector
# Model Used
Model Name : YoloV3 Tiny
# Dataset and Framework links:
https://www.google.com/url?sa=D&q=https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz&ust=1632846780000000&usg=AOvVaw2gPNdNe6SP0ZL0A80j1cfh&hl=en&source=gmail
# Framework Links:
# To Train the model using Yolov3:
https://medium.com/@today.rafi/train-your-own-tiny-yolo-v3-on-google-colaboratory-with-the-custom-dataset-2e35db02bf8f
# To optimize the model using openvino:
https://docs.openvinotoolkit.org/2020.1/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html
# Explanation about Yolov3:
Yolov3 is a realtime object detction framework which got trained on 80 classes on COCO dtaaset.
Transfer learning can be a useful way to quickly retrain YOLOv3 on new data without needing to retrain the entire network.having collected the data and annotated it ,we can download the base weights and we can start our training process from the custom dataset
# Explanation about Openvino(Model Optimization)
openvino is a way of optimizing the model that got trained on any platform like tensorlfow, keras , caffe and getting those source model converted to the Intermediate Representation which runs on the integrated GPU in INTEL with the help of model optimizer. Since it optimizes by combining layers together, Accuracy remains same even after optimization.Got three precisions FP32,FP8, INT8.
# Primary Analysis:
Having got the data, the data has been converted to the format which can be used to train yolov3 tiny . Yolov3 tiny is chosen because the model is already trained on 80 classes and doing transfer learning on the base model would give us more accuracy on limited dataset
# Assumptions:
Fixed SCenario data must be avoided so that our model becomes more generalised and if we need to make the model more generalized we have to collect data from different data sources
We should also analyze the dataset so that the model donot overfit
#Inference:
Once we got the frozen graph by running convert weight.py, then we can convert the pb graph to Intermediate representation by using the modeloptimizer in the openvino deployment tools folder
# False positives 
Tweak the confidence so that false detections with less confidence is neglected
I have also calculated the area for avoiding the detections which includes other than vehicle type car
REtrain the network by removing data that actually confuses the network
# Conclusions
Having trained the model and optimized it to the IR graph which actually runs on the integrated GPU
