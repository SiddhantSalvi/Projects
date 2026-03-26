import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import math
import time
import random
import argparse
import logging
import warnings
import importlib
import collections
from models.diffusion_transformer import DiT, VAE, TextEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import TextImageDataset
from tqdm import tqdm
from torch.utils.data import random_split
from torchvision.utils import save_image

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def load_data(args,text_file, image_dir, transform):
    data = TextImageDataset(text_file=text_file, image_dir=image_dir, transform=transform)
    # split data into train and val
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, 
                            [train_size, val_size], 
                            generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    val_data_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    return train_data_loader, val_data_loader

def get_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT().to(device)
    vae = VAE(device)
    text_encoder = TextEncoder(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        eps=args.epsilon
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=100, 
        gamma=0.1)
    p_uncond = args.p_uncond
    return model, vae, text_encoder, optimizer, scheduler, p_uncond, device

def train(
    model, 
    vae, 
    text_encoder, 
    optimizer, 
    scheduler, 
    train_data_loader, 
    num_epochs, 
    device, 
    p_uncond
    ) -> float:
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_data_loader)):
            images, texts = batch
            batch_size = images.shape[0]
            images = images.to(device)
            # encode image and text
            optimizer.zero_grad()
            encoded_images = vae.encode_latents(images)
            encoded_texts = text_encoder(texts)
            mask = torch.rand(encoded_texts.shape[0], 1).to(device) < p_uncond
            encoded_texts = torch.where(mask, torch.zeros_like(encoded_texts), encoded_texts)
            # sample timestep a scalar in range [0,1]
            timestep = torch.rand(batch_size).to(device)
            x_0 = torch.randn_like(encoded_images)
            timestep_expanded = timestep.view(-1, 1, 1, 1)
            x_t = (1 - timestep_expanded) * x_0 + timestep_expanded * encoded_images
            target = encoded_images - x_0
            pred_v = model(x_t, timestep, encoded_texts)
            loss = F.mse_loss(pred_v, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        sampled_image = one_step_sampling_cfg(model, vae, text_encoder, prompt="a beautiful woman", device=device)
        save_image(sampled_image, "sampled_image.png")
        torch.save(model.state_dict(), f"flow_matching_model_last.pth")
    return avg_loss

def one_step_sampling_cfg(model, vae, text_encoder, prompt, device, steps=20, cfg_scale = 4.0):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(1,4,32,32).to(device)
        encoded_text = text_encoder(prompt)
        encoded_text = encoded_text.to(device)
        uncond_encoded_text = torch.zeros_like(encoded_text).to(device)
        final_text = torch.cat([uncond_encoded_text, encoded_text], dim=0)
        dt = 1.0/steps
        for i in range(steps):
            t = torch.ones(1, device=device) * (i / steps)
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t,t], dim=0)
            v_total = model(x_in, t_in, final_text)
            v_uncond, v_cond = v_total.chunk(2, dim=0)
            v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
            x_t = x_t + dt * v_final
        pred_image = vae.decode_latents(x_t)
        return pred_image

def one_step_sampling_no_cfg(model, vae, text_encoder, prompt, device, steps=20):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(1,4,32,32).to(device)
        encoded_text = text_encoder(prompt)
        encoded_text = encoded_text.to(device)
        dt = 1.0/steps
        for i in range(steps):
            t = torch.ones(1, device=device) * (i / steps)
            pred_v = model(x_t, t, encoded_text)
            x_t = x_t + dt * pred_v
        pred_image = vae.decode_latents(x_t)
        return pred_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    args = parser.parse_args()
    train_data_loader, val_data_loader = load_data(
        args, 
        "flickr30k/captions.txt", 
        "flickr30k/Images", 
        get_transform()
    )
    model, vae, text_encoder, optimizer, scheduler, p_uncond, device = get_model(args)
    avg_loss = train(
        model,
        vae, 
        text_encoder, 
        optimizer, 
        scheduler, 
        train_data_loader,
        args.num_epochs, 
        device, 
        p_uncond
    )