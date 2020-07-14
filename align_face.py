import numpy as np
import PIL
import PIL.Image
import sys
import os
import glob
import scipy
import scipy.ndimage
import dlib
from drive import open_url
from pathlib import Path
import argparse
from bicubic import BicubicDownSample
import torchvision
from shape_predictor import align_face

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='realpics', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input', help='output directory')
parser.add_argument('-output_size', type=int, default=32, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-output_dir2', type=str, default='out', help='output directory')
parser.add_argument('-output_size2', type=int, default=1024, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

args = parser.parse_args()  

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)

print("Downloading Shape Predictor")
f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
predictor = dlib.shape_predictor(f)

for im in Path(args.input_dir).glob("*.*"):
  if str(im) == args.input_dir+'/.ipynb_checkpoints':  # for google colab
    continue
  else:
    faces = align_face(str(im),predictor)

    for i,face in enumerate(faces):  # save pic of size 32*32
        if(args.output_size):
            factor = 1024//args.output_size
            assert args.output_size*factor == 1024
            D = BicubicDownSample(factor=factor)
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
            face_s = torchvision.transforms.ToPILImage()(face_tensor_lr)

        face_s.save(Path(args.output_dir) / (im.stem+'_S.png'))

    for i,face in enumerate(faces):  # save pic of size 1024*1024
        if(args.output_size2):
            factor = 1024//args.output_size2
            assert args.output_size2*factor == 1024
            D = BicubicDownSample(factor=factor)
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
            face_l = torchvision.transforms.ToPILImage()(face_tensor_lr)

        face_l.save(Path(args.output_dir2) / (im.stem+'_L.png'))
        

