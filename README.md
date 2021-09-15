# CPEE_2021_cnn_ensemble_face

Source codes for article on CPEE 2021 conference [https://edison.fel.zcu.cz/html/cpee2021/](https://edison.fel.zcu.cz/html/cpee2021/). The code if fully runnable and can be used to determine detailed parameters used for the training procedure.

## Abstract

The paper considers the problem of recognition of face images using an ensemble of deep CNN networks. The solution combines different feature selection methods and three types of classifiers: support vector machine, a random forest of decision trees, and softmax built into the CNN classifier. Deep learning fulfills an important role in the developed system. The numerical descriptors created in the last locally connected convolutional layer of CNN flattened to the form of a vector, are subject to four different selection mechanisms. Their results are delivered to the three classifiers which are the members of the ensemble. The developed system was tested on the problem of face recognition. The dataset was composed of 68 classes of greyscale images. The results of experiments have shown significant improvement of class recognition resulted from the application of the ensemble.

## Keywords

CNN, ensemble of classifiers, face recognition, feature selection, deep networks.

## Running environment

The source codes were executed on MATLAB version R2021a.

## Directory structure

* `data` - folder with a few examples of data files; due to licensing reasons we are unable to publish the full dataset; you can contact authors: robert.szmurlo@pw.edu.pl, to get the full dataset.;
* `src` - file with full source code compatible with MATLAB R2021a:
  * `alex_transf_learning_act_mod5_only_cnn_softmax.m` -source code for experiment with CNN network and custom deep network classifier layer ending with softmax function,
  * `alex_transf_learning_act_mod5_only_svm.m` - source code with CNN network for feature extraction and SVM as the standalone classifier,
  * `alex_transf_learning_act_mod5_only_treebagger.m` - source code with CNN network for feature extraction and Random forest trees as the standalone classifier,
  * `alex_transf_learning_act_mod5_softmax_svm.m` - source code with CNN network for feature extraction and SVM accompanied with deep network softmax classifier,
  * `alex_transf_learning_act_mod5_softmax_svm_treebagger.m` - source code with CNN network for feature extraction and SVM accompanied with deep network softmax, and random forest of decission  trees classifiers,
  * `alex_transf_learning_act_mod5_softmax_svm.m` - source code with CNN network for feature extraction and SVM accompanied with random forest decision trees classifier,
  * `customreader.m` - custom images loading function for MATLAB dataset function.
