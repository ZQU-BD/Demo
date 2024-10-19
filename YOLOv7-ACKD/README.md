# Attention-based and Context-aware Knowledge Distillation

Knowledge Distillation for Object Detection
## 👋 Introduction
This project is about agricultural disease detection. We have incorporated several intermediate feature layer distillation methods into the YOLOv7 framework. The code for the knowledge distillation methods can be found at the bottom of utils/loss.py.

We propose a new method called Attention-based and Context-aware Knowledge Distillation (ACKD), which utilizes gradient information to measure the importance of features, enabling the student model to focus on more valuable knowledge related to the final detection.
