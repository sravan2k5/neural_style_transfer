#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import streamlit as st

st.markdown("""
    <style>
    .stApp {
        background-color: #ffe6ea;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Neural Style Transfer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image, imsize):
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def im_convert(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    return transforms.ToPILImage()(image)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

def get_model_and_losses(style_img, content_img):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = transforms.Normalize(mean, std)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

def run_style_transfer(_, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_model_and_losses(style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            return style_score + content_score
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

content_file = st.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'])
style_file = st.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'])

if content_file and style_file:
    with st.spinner("Applying Style Transfer... Please wait..."):
        imsize = 512 if torch.cuda.is_available() else 256

        content_image = Image.open(content_file).convert("RGB")
        style_image = Image.open(style_file).convert("RGB")
        style_image = style_image.resize(content_image.size)

        content_tensor = image_loader(content_image, content_image.size)
        style_tensor = image_loader(style_image, content_image.size)
        input_img = content_tensor.clone()

        output = run_style_transfer(None, content_tensor, style_tensor, input_img)

        st.success("Style Transfer Complete!")

        st.subheader("Content Image")
        st.image(content_image, width=300)

        st.subheader("Style Image")
        st.image(style_image, width=300)

        st.subheader("Stylized Image")
        st.image(im_convert(output), width=812)


# In[ ]:




