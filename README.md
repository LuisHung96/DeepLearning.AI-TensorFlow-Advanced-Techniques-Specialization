# <b>DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization<b>

[![Made with Google Colab](https://img.shields.io/badge/Made%20with-Google%20Colab-yellow?style=for-the-badge&logo=Google%20Colab)](https://colab.research.google.com/) [![Made with Tensorflow](https://img.shields.io/badge/Framework-TensorFlow-green)](https://www.tensorflow.org/) ![Crates.io](https://img.shields.io/crates/l/rustc-serialize?style=flat-square) [![GitHub last commit](https://img.shields.io/github/last-commit/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization?color=red)](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization)

![](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/images/course.jpg)

### Instructed by [<img src="https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/images/laurence_moroney.png" width="20"/> Laurence Moroney](https://laurencemoroney.com/about.html)
### Offered by [<img src="https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/images/deeplearning_logo.png" width="200"/>](https://www.deeplearning.ai)
- ---

## <b>About this Specialization<b>

Welcome! This repository contains the projects and exercises I completed during the **_TensorFlow: Advanced Techniques Specialization_** by **_DeepLearning.AI_** on Coursera.
This advanced program strengthened my expertise in **_TensorFlow 2.x_** and **_Keras_**, providing me with the skills to design, build, and optimize sophisticated deep learning models. Through hands-on projects, I explored:

1. Advanced Model Architectures & Customization:
    * Built non-sequential models using the Functional API, implementing custom layers, loss functions, and multi-input/output workflows.
    * Extracted features and performed fine-tuning on pre-trained models like _VGG16/19_, _ResNet50_ and _InceptionV3_ for custom tasks.
    * Built _FCN-8_ and _U-Net_ from scratch, and fine-tuned _RetinaNet_ & _Faster R-CNN_ using the _TensorFlow Object Detection API_.
    * Applied transfer learning with _MobileNetV2_ for image classification tasks using TensorFlow.

2. Distributed Training Strategies:
    *	```tf.distribute.MirroredStrategy``` for synchronous training on multiple _GPUs_ on a single machine, efficiently replicating the model and synchronizing gradients.
    *	```tf.distribute.experimental.TPUStrategy```  to accelerate and scale model training on _TPUs_ for maximum performance.
    *	```tf.distribute.OneDeviceStrategy``` for debugging and baseline performance on a single device (_CPU/GPU_).

3. Advanced Computer Vision & Evaluation:
    *	Gained practical experience in key CV tasks: object detection, image segmentation (semantic and instance), and model interpretability using _Grad-CAM_.
    * Moved beyond basic accuracy; evaluated segmentation performance using standard metrics like Intersection over Union (_IoU_) and _Dice Coefficient_.

4. Generative Deep Learning:
    * Developed and trained Variational Autoencoders (_VAEs_) for image generation and latent space exploration.
    * Built and tuned Generative Adversarial Networks (_GANs_), including _DCGAN_ for simple generation and _CycleGAN_ for unpaired image-to-image translation.
    * Implemented Neural Style Transfer to combine the content and style of different images.

5. Low-Level Control & Optimization: 
    * Implemented custom training loops using ```tf.GradientTape``` for maximum flexibility over the training process.

Completing this specialization empowered me to tackle real-world challenges—from building exotic model topologies to deploying scalable AI solutions. The projects here reflect my journey through advanced TensorFlow capabilities, blending theory with practical implementation.

**Feel free to explore the assignments, labs and quizzes I’ve shared. Hope you find them useful!**

Please, check **<i>[Coursera Honor Code](https://www.coursera.support/s/article/209818863-Coursera-Honor-Code?language=en_US)</i>** before you take a look at the assignments.

For more you can check **[course info](https://www.deeplearning.ai/courses/tensorflow-advanced-techniques-specialization/)**.

![DeepLearning AI TensorFlow Advanced Techniques Specialization](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/certificates/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization_LH.jpg)

- ---
## <b>Courses and Certificates<b>
  - ### <b>[Course 1 - Custom Models, Layers, and Loss Functions with TensorFlow](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C1)<b>
    * <b>[Week 1 - Functional APIs](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C1/W1)</b>
      
      * <b>Assignment:<b>

        * <b>Multiple Output Models using Keras Functional API:<b> <b>[_C1W1_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W1/Assignment/C1W1_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Functional API Practice:</b> <b>[_C1_W1_Lab_1_functional-practice.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W1/Ungraded/Lab1/C1_W1_Lab_1_functional-practice.ipynb)<b>
        * <b>Multi-output:</b> <b>[_C1_W1_Lab_2_multi-output.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W1/Ungraded/Lab2/C1_W1_Lab_2_multi-output.ipynb)<b>
        * <b>Siamese network:</b> <b>[_C1_W1_Lab_3_siamese-network.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W1/Ungraded/Lab3/C1_W1_Lab_3_siamese-network.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W1/Quiz.png)<b>
        
    * <b>[Week 2 - Custom Loss Functions](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C1/W2)</b>

      * <b>Assignment:<b>
        * <b>Creating a custom loss function:<b> <b>[_C1W2_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W2/Assignment/C1W2_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Huber Loss lab:</b> <b>[_C1_W2_Lab_1_huber-loss.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W2/Ungraded/Lab1/C1_W2_Lab_1_huber-loss.ipynb)<b>
        * <b>Huber Loss object:</b> <b>[_C1_W2_Lab_2_huber-object-loss.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W2/Ungraded/Lab2/C1_W2_Lab_2_huber-object-loss.ipynb)<b>
        * <b>Contrastive loss in the siamese network:</b> <b>[_C1_W1_Lab_3_siamese-network.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W2/Ungraded/Lab3/C1_W1_Lab_3_siamese-network.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W2/Quiz.png)<b>

    * <b>[Week 3 - Custom Layers](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C1/W3)</b>

      * <b>Assignment:<b>
        * <b>Implement a Quadratic Layer:<b> <b>[_C1W3_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W3/Assignment/C1W3_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Lambda layer:</b> <b>[_C1_W3_Lab_1_lambda-layer.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W3/Ungraded/Lab1/C1_W3_Lab_1_lambda-layer.ipynb)<b>
        * <b>Custom dense layer:</b> <b>[_C1_W3_Lab_2_custom-dense-layer.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W3/Ungraded/Lab2/C1_W3_Lab_2_custom-dense-layer.ipynb)<b>
        * <b>Activation in a custom layer:</b> <b>[_C1_W3_Lab_3_custom-layer-activation.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W3/Ungraded/Lab3/C1_W3_Lab_3_custom-layer-activation.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W3/Quiz.png)<b>

    * <b>[Week 4 - Custom Models](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C1/W4)</b>
      
      * <b>Assignment:<b>
        * <b>Create a VGG network:<b> <b>[_C1W4_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W4/Assignment/C1W4_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Build a basic model:</b> <b>[_C1_W4_Lab_1_basic-model.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W4/Ungraded/Lab1/C1_W4_Lab_1_basic-model.ipynb)<b>
        * <b>Build a ResNet model:</b> <b>[_C1_W4_Lab_2_resnet-example.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W4/Ungraded/Lab2/C1_W4_Lab_2_resnet-example.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W4/Quiz.png)<b>

    * <b>[Week 5 - Bonus Content (Callbacks)](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C1/W5/Ungraded)</b>

      * <b>Ungraded Labs:<b>

        * <b>Built-in Callbacks:</b> <b>[_C1_W5_Lab_1_exploring-callbacks.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W5/Ungraded/Lab1/C1_W5_Lab_1_exploring-callbacks.ipynb)<b>
        * <b>Custom Callbacks:</b> <b>[_C1_W5_Lab_2_custom-callbacks.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C1/W5/Ungraded/Lab2/C1_W5_Lab_2_custom-callbacks.ipynb)<b>
      
    <details>
      <summary>Certificate</summary>
           <img src="https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/certificates/C1_certificate_LH.jpg" alt="TensorFlow C1 Certificate">
    </details>
  - ---

  - ### <b>[Course 2 - Custom and Distributed Training with TensorFlow](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C2)<b>
    * <b>[Week 1 - Differentiation and Gradients](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C2/W1)</b>
      
      * <b>Assignment:<b>

        * <b>Basic Tensor Operations:<b> <b>[_C2W1_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W1/Assignment/C2W1_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Basic Tensors:</b> <b>[_C2_W1_Lab_1_basic-tensors.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W1/Ungraded/Lab1/C2_W1_Lab_1_basic-tensors.ipynb)<b>
        * <b>Gradient Tape Basics:</b> <b>[_C2_W1_Lab_2_gradient-tape-basics.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W1/Ungraded/Lab2/C2_W1_Lab_2_gradient-tape-basics.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W1/Quiz.png)<b>
        
    * <b>[Week 2 - Custom Training](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C2/W2)</b>

      * <b>Assignment:<b>
        * <b>Breast Cancer Prediction:<b> <b>[_C2W2_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W2/Assignment/C2W2_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Training Basics:</b> <b>[_C2_W2_Lab_1_training-basics.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W2/Ungraded/Lab1/C2_W2_Lab_1_training-basics.ipynb)<b>
        * <b>Fashion MNIST using Custom Training Loop:</b> <b>[_C2_W2_Lab_2_training-categorical.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W2/Ungraded/Lab2/C2_W2_Lab_2_training-categorical.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W2/Quiz.png)<b>

    * <b>[Week 3 - Graph Mode](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C2/W3)</b>

      * <b>Assignment:<b>
        * <b>Horse or Human?:<b> <b>[_C2W3_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W3/Assignment/C2W3_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>AutoGraph Basics:</b> <b>[_C2_W3_Lab_1_autograph-basics.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W3/Ungraded/Lab1/C2_W3_Lab_1_autograph-basics.ipynb)<b>
        * <b>AutoGraph:</b> <b>[_C2_W3_Lab_2-graphs-for-complex-code.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W3/Ungraded/Lab2/C2_W3_Lab_2-graphs-for-complex-code.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W3/Quiz.png)<b>

    * <b>[Week 4 - Distributed Training](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C2/W4)</b>
      
      * <b>Assignment:<b>
        * <b>Distributed Strategy:<b> <b>[_C2W4_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W4/Assignment/C2W4_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Mirrored Strategy:</b> <b>[_C2_W4_Lab_1_basic-mirrored-strategy.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W4/Ungraded/Lab1/C2_W4_Lab_1_basic-mirrored-strategy.ipynb)<b>
        * <b>Multi GPU Mirrored Strategy:</b> <b>[_C2_W4_Lab_2_multi-GPU-mirrored-strategy.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W4/Ungraded/Lab2/C2_W4_Lab_2_multi-GPU-mirrored-strategy.ipynb)<b>
        * <b>TPU Strategy:</b> <b>[_C2_W4_Lab_3_using-TPU-strategy.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W4/Ungraded/Lab3/C2_W4_Lab_3_using-TPU-strategy.ipynb)<b>
        * <b>One Device Strategy:</b> <b>[_C2_W4_Lab_4_one-device-strategy.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W4/Ungraded/Lab4/C2_W4_Lab_4_one-device-strategy.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C2/W4/Quiz.png)<b>
   
    <details>
      <summary>Certificate</summary>
           <img src="https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/certificates/C2_certificate_LH.jpg" alt="TensorFlow C2 Certificate">
    </details>
  - ---

  - ### <b>[Course 3 - Advanced Computer Vision with TensorFlow](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C3)<b> 
    * <b>[Week 1 - Introduction to Computer Vision](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C3/W1)</b>
      
      * <b>Assignment:<b>

        * <b>Bird Boxes:<b> <b>[_C3W1_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W1/Assignment/C3W1_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Transfer Learning:</b> <b>[_C3_W1_Lab_1_transfer_learning_cats_dogs.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W1/Ungraded/Lab1/C3_W1_Lab_1_transfer_learning_cats_dogs.ipynb)<b>
        * <b>Transfer Learning with ResNet 50:</b> <b>[_C3_W1_Lab_2_Transfer_Learning_CIFAR_10.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W1/Ungraded/Lab2/C3_W1_Lab_2_Transfer_Learning_CIFAR_10.ipynb)<b>
        * <b>Image Classification and Object Localization:</b> <b>[_C3_W1_Lab_3_Object_Localization.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W1/Ungraded/Lab3/C3_W1_Lab_3_Object_Localization.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W1/Quiz.png)<b>
        
    * <b>[Week 2 - Object Detection](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C3/W2)</b>

      * <b>Assignment:<b>
        * <b>Zombie Detector:<b> <b>[_C3W2_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W2/Assignment/C3W2_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Implement Simple Object Detection:</b> <b>[_C3_W2_Lab_1_Simple_Object_Detection.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W2/Ungraded/Lab1/C3_W2_Lab_1_Simple_Object_Detection.ipynb)<b>
        * <b>Predicting Bounding Boxes for Object Detection:</b> <b>[_C3_W2_Lab_2_Object_Detection.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W2/Ungraded/Lab2/C3_W2_Lab_2_Object_Detection.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W2/Quiz.png)<b>

    * <b>[Week 3 - Image Segmentation](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C3/W3)</b>

      * <b>Assignment:<b>
        * <b>Image Segmentation of Handwritten Digits:<b> <b>[_C3W3_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W3/Assignment/C3W3_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Implement a Fully Convolutional Neural Network:</b> <b>[_C3_W3_Lab_1_VGG16-FCN8-CamVid.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W3/Ungraded/C3_W3_Lab_1_VGG16-FCN8-CamVid.ipynb)<b>
        * <b>Implement a UNet:</b> <b>[_C3_W3_Lab_2_OxfordPets-UNet.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W3/Ungraded/C3_W3_Lab_2_OxfordPets-UNet.ipynb)<b>
        * <b>Instance Segmentation Demo:</b> <b>[_C3_W3_Lab_3_Mask-RCNN-ImageSegmentation.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W3/Ungraded/C3_W3_Lab_3_Mask-RCNN-ImageSegmentation.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W3/Quiz.png)<b>

    * <b>[Week 4 - Visualization and Interpretability](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C3/W4)</b>
      
      * <b>Assignment:<b>
        * <b>Cats vs Dogs Saliency Maps:<b> <b>[_C3W4_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W4/Assignment/C3W4_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Class Activation Maps with Fashion MNIST:</b> <b>[_C3_W4_Lab_1_FashionMNIST-CAM.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W4/Ungraded/C3_W4_Lab_1_FashionMNIST-CAM.ipynb)<b>
        * <b>Class Activation Maps "Cats vs Dogs":</b> <b>[_C3_W4_Lab_2_CatsDogs-CAM.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W4/Ungraded/C3_W4_Lab_2_CatsDogs-CAM.ipynb)<b>
        * <b>Saliency Maps:</b> <b>[_C3_W4_Lab_3_Saliency.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W4/Ungraded/C3_W4_Lab_3_Saliency.ipynb)<b>
        * <b>GradCAM:</b> <b>[_C3_W4_Lab_4_GradCam.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W4/Ungraded/C3_W4_Lab_4_GradCam.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C3/W4/Quiz.png)<b>
   
    <details>
      <summary>Certificate</summary>
          <img src="https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/certificates/C3_certificate_LH.jpg" alt="TensorFlow C3 Certificate">
    </details>
  - ---

  - ### <b>[Course 4 - Generative Deep Learning with TersorFlow](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C4)<b> 
    * <b>[Week 1 - Style Transfer](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C4/W1)</b>
      
      * <b>Assignment:<b>

        * <b>Style Transfer Dog:<b> <b>[_C4W1_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W1/Assignment/C4W1_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>Neural Style Transfer:</b> <b>[_C4_W1_Lab_1_Neural_Style_Transfer.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W1/Ungraded/C4_W1_Lab_1_Neural_Style_Transfer.ipynb)<b>
        * <b>Fast Neural Style Transfer:</b> <b>[_C4_W1_Lab_2_Fast_NST.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W1/Ungraded/C4_W1_Lab_2_Fast_NST.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W1/Quiz.png)<b>
        
    * <b>[Week 2 - AutoEncoders](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C4/W2)</b>

      * <b>Assignment:<b>
        * <b>AutoEncoder Model Loss and Accuracy:<b> <b>[_C4W2_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Assignment/C4W2_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>First Autoencoder:</b> <b>[_C4_W2_Lab_1_FirstAutoEncoder.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Ungraded/C4_W2_Lab_1_FirstAutoEncoder.ipynb)<b>
        * <b>MNIST AutoEncoder:</b> <b>[_C4_W2_Lab_2_MNIST_Autoencoder.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Ungraded/C4_W2_Lab_2_MNIST_Autoencoder.ipynb)<b>
        * <b>MNIST Deep AutoEncoder:</b> <b>[_C4_W2_Lab_3_MNIST_DeepAutoencoder.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Ungraded/C4_W2_Lab_3_MNIST_DeepAutoencoder.ipynb)<b>
        * <b>Fashion MNIST - CNN AutoEncoder:</b> <b>[_C4_W2_Lab_4_FashionMNIST_CNNAutoEncoder.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Ungraded/C4_W2_Lab_4_FashionMNIST_CNNAutoEncoder.ipynb)<b>
        * <b>Fashion MNIST - Noisy CNN AutoEncoder:</b> <b>[_C4_W2_Lab_5_FashionMNIST_NoisyCNNAutoEncoder.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Ungraded/C4_W2_Lab_5_FashionMNIST_NoisyCNNAutoEncoder.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W2/Quiz.png)<b>

    * <b>[Week 3 - Variational AutoEncoders](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C4/W3)</b>

      * <b>Assignment:<b>
        * <b>Anime Faces:<b> <b>[_C4W3_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W3/Assignment/C4W3_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>MNIST Variational AutoEncoder:</b> <b>[_C4_W3_Lab_1_VAE_MNIST.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W3/Ungraded/C4_W3_Lab_1_VAE_MNIST.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W3/Quiz.png)<b>

    * <b>[Week 4 - GANs](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/tree/main/C4/W4)</b>
      
      * <b>Assignment:<b>
        * <b>Generated Hands:<b> <b>[_C4W4_Assignment.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W4/Assignment/C4W4_Assignment.ipynb)<b>

      * <b>Ungraded Labs:<b>

        * <b>First GAN:</b> <b>[_C4_W4_Lab_1_First_GAN.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W4/Ungraded/C4_W4_Lab_1_First_GAN.ipynb)<b>
        * <b>First DCGAN:</b> <b>[_C4_W4_Lab_2_First_DCGAN.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W4/Ungraded/C4_W4_Lab_2_First_DCGAN.ipynb)<b>
        * <b>CelebA GAN Experiments:</b> <b>[_C4_W4_Lab_3_CelebA_GAN_Experiments.ipynb_](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W4/Ungraded/C4_W4_Lab_3_CelebA_GAN_Experiments.ipynb)<b>

      * <b>Quiz:<b>
        * <b>[Quiz](https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/C4/W4/Quiz.png)<b>
    <details>
      <summary>Certificate</summary>
            <img src="https://github.com/LuisHung96/DeepLearning.AI-TensorFlow-Advanced-Techniques-Specialization/blob/main/resources/certificates/C4_certificate_LH.jpg" alt="TensorFlow C4 Certificate">
    </details>
  - ---

## <b>Reference<b>
<b>[DeepLearning.AI TensorFlow Advanced Techniques Specialization](https://www.coursera.org/specializations/tensorflow-advanced-techniques)<b>
