# Comparative-Analysis-of-VGG16-and-ResNet50-Architectures-on-CIFAR-10-Image-Classification
Comparative Analysis of VGG16 and ResNet50 Architectures on CIFAR-10 Image Classification
________________________________________
Team Details: (Computer Science Engineer)
1. Pushkarjay Ajay  
2. Bhavya Singh
3. Ayush Kumar
4. Satyam
5. Eshan Ghoshrave
6. Ajitab Mayank
Under the guidance of:
Dr. Minakhi Rout (Associate Professor, School of Computer Engineering.)
KIIT University, Bhubaneswar, India
________________________________________
Abstract
This study evaluates the performance of two prominent convolutional neural network (CNN) architectures, VGG16 and ResNet50, on the CIFAR-10 dataset. Despite ResNet50's deeper architecture and advanced design, our experiments reveal that VGG16 outperforms ResNet50 in terms of accuracy. We analyze the models' performance in accuracy, training time, inference time, and computational complexity, highlighting key insights and areas for improvement. The study also provides a visual comparison through confusion matrices, training history, and per-class performance analysis.
Keywords: VGG16, ResNet50, CIFAR-10, Image Classification, Deep Learning
________________________________________
Table of Contents
1.	Introduction
2.	Related Work
3.	Methodology 
o	Dataset
o	Model Architectures
o	Training Configuration
4.	Experimental Implementation
5.	Results and Analysis 
o	Performance Metrics
o	Training History
o	Confusion Matrices
o	Performance Comparison
o	Per-Class Performance
o	Observations
6.	Discussion
7.	Conclusion
8.	References
9.	Appendix (Includes Code)
________________________________________
1. Introduction
Deep learning has significantly advanced image classification, with CNN architectures like VGG16 and ResNet50 playing a crucial role. VGG16 is known for its simplicity and uniform structure, whereas ResNet50 introduces residual connections to overcome vanishing gradient issues. This study compares the effectiveness of these models in classifying images from CIFAR-10, a commonly used benchmark dataset. (scitepress) (SSRN)
________________________________________
2. Related Work
•	He et al. introduced ResNet architectures, achieving state-of-the-art performance on CIFAR-10 (CV Foundation).
•	Transfer learning significantly improved ResNet50’s performance on CIFAR-10, reaching 92% accuracy (DiVA Portal).
•	A comparative study between VGG16 and other CNNs showed varying performance based on dataset complexity (SSRN).
________________________________________
3. Methodology
3.1. Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images categorized into 10 classes. It is divided into 50,000 training and 10,000 testing images (DiVA Portal) (SSRN) (CV Foundation).
3.2. Model Architectures
•	VGG16: A 16-layer network primarily using 3x3 convolutional layers with fully connected layers at the end.
•	ResNet50: A 50-layer deep residual network employing skip connections for improved gradient flow. (CV Foundation)
3.3. Training Configuration
•	Preprocessing: Resizing images to 128x128 pixels.
•	Hyperparameters: 
o	Epochs: 10
o	Batch Size: 64
o	Optimizer: Adam (learning rate = 0.001)
________________________________________
4. Experimental Implementation
The models were trained on CIFAR-10 using a batch size of 64 for 10 epochs. The implementation utilized TensorFlow and Keras, with training conducted on a GPU-enabled system for efficiency.
Refer to Appendix-B

________________________________________
5. Results and Analysis
5.1. Performance Metrics
Metric	VGG16	ResNet50
Accuracy	70.59%	34.46%
Training Time	372.16s	297.74s
Inference Time	155.99ms	102.08ms
Total Parameters	14.98M	24.64M
Trainable Params	0.27M	1.05M
5.2. Training History
 
5.3. Confusion Matrices
 
5.4. Performance Comparison
 
5.5. Per-Class Performance
 
5.6. Observations
•  Accuracy: VGG16 achieved a higher accuracy compared to ResNet50. 
•  Precision and Recall: Both models recorded 0% precision and recall, indicating potential issues in model training or evaluation. 
•  Training and Inference Time: ResNet50 was faster in both training and inference, likely due to its architectural optimizations. 
•  Model Complexity: ResNet50 has more parameters, suggesting a higher capacity but also a greater risk of overfitting.
________________________________________
6. Discussion
•	Training Duration: A 10-epoch period may not be optimal for ResNet50 to fully converge.
•	Hyperparameter Tuning: Adjusting learning rates or optimizers could improve performance.
•	Data Augmentation: Increasing augmentation techniques might enhance ResNet50’s accuracy.
________________________________________
7. Conclusion
This study shows that VGG16 outperforms ResNet50 under the given experimental setup. Future work should explore deeper training cycles, improved hyperparameter tuning, and additional data preprocessing techniques to optimize performance.
________________________________________
8. References
1.	He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. (CV Foundation)
2.	Adhikari, T. (2023). Designing a Convolutional Neural Network for Image Recognition: A Comparative Study of Different Architectures and Training Techniques. SSRN. (SSRN)
3.	Evaluating Transfer Learning Capabilities of Neural Networks on CIFAR-10. DiVA Portal. (DiVA Portal)
4.	CIFAR-10 Benchmark (Image Classification). Papers With Code. (Papers With Code)
5.	Comparative analysis of various models for image classification on Cifar-100 dataset. (ResearchGate)
________________________________________
9. Appendix
	Appendix-A
	vgg16_resnet50_comparison.pdf
Appendix-B
Github/Pushkarjay/Code.py

