#!/usr/bin/env python
# coding: utf-8

# # AUTOANNOTATOR v0.4
# Jiri Podivin - 2023

import numpy as np
from torchvision.ops import masks_to_boxes
import argparse
from transformers import CLIPProcessor, CLIPModel, pipeline
import torch

from PIL import Image
import os
import glob
import re
import json

DEFAULT_CATEGORIES = [
    {
        "label":"foliage",
        "clip_desc":"photo of a plant leaf",
        "shortest_side": 100
    },
    {
        "label":"fruit",
        "clip_desc":"photo of a plant fruit",
        "shortest_side": 100
    },
    {
        "label":"stem",
        "clip_desc":"photo of a plant stem, trunk or branch",
        "shortest_side": 100
    },
    {
        "label":"flower",
        "clip_desc":"photo of a plant flower",
        "shortest_side": 100
    },
]

def get_labels(
        source_path, labels=None, min_certainty=0.89, output_dir='./outputs',
        sam_model="facebook/sam-vit-base", clip_model="openai/clip-vit-base-patch32",
        longest_edge=900, device='auto'):
    
    if device == 'auto':
        sam_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clip_device = sam_device
    elif device == 'sam-cpu-clip-gpu':
        sam_device = 'cpu'
        clip_device = 'gpu'
    else:
        sam_device = clip_device = device

    if not labels:
        categories = DEFAULT_CATEGORIES
    else:
        with open(labels, 'r') as f:
            categories = json.load(fp=f)

    paths = [(cat['label'], os.path.join(output_dir, re.sub(r'\W','_', cat['label']))) for cat in categories]
    paths = dict(paths)
    b_box_dir = os.path.join(output_dir, 'b_boxes')
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("Output dir already exists!")
    try:
        os.mkdir(b_box_dir)
    except FileExistsError:
        print("Bounding box dir already exists!")
    for path in paths.values():
        try:
            os.mkdir(path)
        except FileExistsError:
            continue

    ## Segmentation - SAM
    generator = pipeline(
        "mask-generation", device=sam_device, model=sam_model)
    # CLIP
    model = CLIPModel.from_pretrained(clip_model)
    model.to(clip_device)

    for image_path in glob.glob(os.path.join(source_path, "**.jpg")):
        image = Image.open(image_path)
        ratio = longest_edge/max(image.size)
    
        new_size = (
            int(image.size[0]*ratio),
            int(image.size[1]*ratio))

        print(f"image size {image.size} to {new_size}")
        mask_image = image.resize(new_size)
            
        masks = generator(mask_image, points_per_batch=64, stability_score_thresh=0.89)

        # Get BBs
        boxes = masks_to_boxes(torch.tensor(np.array(masks['masks']))) * (1/ratio)
        boxes = np.floor(boxes.numpy()).tolist()

        image_id = os.path.basename(image_path)
        with open(os.path.join(b_box_dir, image_id) + '.json', 'w') as f:
            
            json.dump(
                {
                    "boxes": boxes,
                    "labels": np.zeros(len(boxes)).tolist(),
                    "image_id": image_id}, f)

        sub_images = []

        for box in boxes:
            new_image = image.crop(np.array(box))
            sub_images.append(new_image)


        processor = CLIPProcessor.from_pretrained(clip_model)

        inputs = processor(
            text=[cat['clip_desc'] for cat in categories],
            images=sub_images, return_tensors="pt", padding=True).to(clip_device)

        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        probs = probs.cpu().detach().numpy()

        for img in range(len(sub_images)):
            if np.max(probs[img]) > min_certainty:
                for cat in range(len(categories)):
                    if np.argmax(probs[img]) == cat and categories[cat]['shortest_side'] <= max(sub_images[img].size):
                        print(f"It's a {categories[cat]['label']} size {sub_images[img].size}!")
                        sub_images[img].save(os.path.join(paths[categories[cat]['label']], f"{image_id}_{img:06}.jpg"))


def main():
    parser = argparse.ArgumentParser(
        prog="autoannotator", usage="point to stuff get your things",)
    
    parser.add_argument("source_path", action='store')
    parser.add_argument("--labels", type=str)
    parser.add_argument("--min-certainty", type=float, default=0.89)
    parser.add_argument("--sam-model", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--longest-edge", type=int, default=900, help="Useful when you don't have enough memory for large images.")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu', 'auto', 'sam-cpu-clip-gpu'], default='gpu')
    args = parser.parse_args()
    if args.min_certainty > 1.0 or args.min_certainty < 0.0:
        raise ValueError("min-certainty must fall in the <0.0,1.0> range.")
    get_labels(
        args.source_path, args.labels, args.min_certainty,
        sam_model=args.sam_model, clip_model=args.clip_model,
        longest_edge=args.longest_edge)

if __name__ == '__main__':
    main()