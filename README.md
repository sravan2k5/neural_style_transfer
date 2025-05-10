# neural_style_transfer
Neural Style Transfer with PyTorch
This project implements Neural Style Transfer (NST) using PyTorch to blend the content of one image with the artistic style of another. It uses a pre-trained VGG19 model to recreate a new image that combines both sources — generating AI-based artwork from photography and painting.

🧠 What is Neural Style Transfer?
Neural Style Transfer is a deep learning technique where a content image is "repainted" in the style of a different image. For example, you can make a selfie look like a Van Gogh painting.

🚀 Features
Transfer artistic style from a painting to a photograph

Uses pre-trained VGG19 convolutional neural network

Supports custom content and style images

Adjustable style/content weights and number of optimization steps

Output saved as a new image

🛠️ Technologies Used
Python 3.x

PyTorch

PIL (Python Imaging Library)

Matplotlib

Torchvision (for pre-trained models)

📦 Installation
Install dependencies using pip:

bash

pip install torch torchvision pillow matplotlib
