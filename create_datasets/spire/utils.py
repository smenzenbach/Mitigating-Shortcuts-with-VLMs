import tqdm
import json
import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image as Image
import os


def remove_object(image, mask):
    """
    Remove an object from an image using a segmentation mask.

    Args:
        image (PIL.Image): Input image.
        mask (np.array): Binary mask with values 0 or 255.

    Returns:
        PIL.Image: Modified image with the object removed.
    """
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Find bounding box coordinates of the mask (where mask is 255)
    y_indices, x_indices = np.where(mask == 255)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return image  # If no object is found, return original image

    # Get bounding box coordinates
    xmin, xmax = x_indices.min(), x_indices.max()
    ymin, ymax = y_indices.min(), y_indices.max()

    # Define the fill color
    fill_color = np.array([124, 116, 104], dtype=np.uint8)

    # Replace pixels inside bounding box with fill color
    image_np[ymin:ymax+1, xmin:xmax+1] = fill_color

    # Convert back to PIL Image
    return Image.fromarray(image_np)

import numpy as np
from PIL import Image

def add_object(image, reference_mask, reference_image):
    """
    Adds an object from the reference image into the target image using a reference mask.

    Args:
        image (PIL.Image): The target image where the object will be added.
        reference_mask (np.array): Binary mask (0 or 255) defining the object in reference_image.
        reference_image (PIL.Image): Image containing the object.

    Returns:
        PIL.Image: Image with the object added.
    """
    # Convert images to NumPy arrays
    image_np = np.array(image)
    reference_image_np = np.array(reference_image)

    # Ensure the reference mask is binary (0 or 1)
    reference_mask = (reference_mask > 0).astype(np.uint8)

    # Expand the mask to 3 channels for RGB images
    mask_3ch = np.stack([reference_mask] * 3, axis=-1)

    # Copy object from reference image where mask == 1
    image_np = np.where(mask_3ch == 1, reference_image_np, image_np)

    # Convert back to PIL Image
    return Image.fromarray(image_np)


def get_groups(metadata):
    """
    Extract groups and group_sizes from the training data and provide lists with filenames for each group.
    The output is a dict containing the following groups:
        'class': images containing the target class
        'confounded': images containing a confounder
        'neither': images containing neither the target class nor a confounder
        'both': images containing both the target class and a confounder
        'just_class': images containing the target class but no confounder
        'just_confounded': images containing a confounder but not the target class
    """

    groups = {}
    groups['class'] = []
    groups['confounded'] = []
    groups['neither'] = []
    groups['both'] = []
    groups['just_class'] = []
    groups['just_confounded'] = []
    
    for i, row in tqdm(metadata.iterrows()):
        # ignore test and validation data
        if row['split'] !=  0:
            continue
        
        if row['class_label'] == 1:
            # it is definitiv an image of the class
            groups['class'].append(row['filename'])
            if row['confounder_label'] == 1:
                # class and confounder
                groups['confounded'].append(row['filename'])
                groups['both'].append(row['filename'])
            else:
                # only class, no confounder
                groups['just_class'].append(row['filename'])
        else:
            if row['confounder_label'] == 1:
                # only confounder
                groups['confounded'].append(row['filename'])
                groups['just_confounded'].append(row['filename'])
            else:
                # neither
                groups['neither'].append(row['filename'])

    return groups

# TODO add_object function, what parameters?

def compute_augmentations(groups):
    """
    Compute the sampling probabilities for the different data augmentation via Spire.
    Provides a sampling probability for each of the following augmentation settings:
        B2M: Removes the confounder from images with both class and confounder (Removal)
        B2S: Removes the class from images with both class and confounder (Removal)
        M2N: Removes the class from images with only the class (Removal)
        S2N: Removes the confounder from images with only the confounder (Removal)
        N2M: Adds the class to images with neither class nor confounder (Addition)
        N2S: Adds the confounder to images with neither class nor confounder (Addition)
        M2B: Adds the confounder to images with only the class (Addition)
        S2B: Adds the class to images with only the confounder (Addition)

    Returns a dict with the respective keys and the corresponding sampling probabilities.
    """

    CHARS = ['B', 'M', 'S', 'N']
    # helper functions
    def unpack_key(key):
        split = key.split('2')
        source = split[0]
        dest = split[1]
        return source, dest

    def get_totals(augs):
        totals = {}
        for c in CHARS:
            totals[c] = torch.tensor(0.0)
        for key in augs:
            source, dest = unpack_key(key)
            totals[dest] += augs[key]    
        return totals

    def eval_augs(sizes, deltas, augs,
                lambda_size = 100.0, lambda_diff = 10.0, lambda_remove = 1.0, lambda_add = 1.0,
                verbose = False):
                
        if verbose:
            print('Augs')
            for key in augs:
                print(key, format(augs[key]))

        cost = torch.tensor(0.0)
        
        # Penalize any illegal sizes
        for key in augs:
            source, dest = unpack_key(key)
            dif = augs[key] - sizes[source]
            cost += lambda_size * dif.clamp_(min = 0.0)
            
        # Penalize errors in meeting the deltas
        totals = get_totals(augs)
        if verbose:
            print('Totals')
            for c in CHARS:
                print(c, format(totals[c]))
            
        for c in CHARS:
            #if c in ['B', 'M']: # Auto-v2 differs from Auto-v1 by this single line
            cost += lambda_diff * torch.abs(totals[c] - deltas[c])
            
        # Check that the intervention does not leave traces
        v = augs['B2M'] - (augs['B2S'] + augs['M2N'] + augs['S2N'])
        if verbose:
            print('Removing Bias', format(v))
        cost += lambda_remove * torch.abs(v)
        
        v = augs['N2S'] - (augs['N2M'] + augs['M2B'] + augs['S2B'])
        if verbose:
            print('Adding Bias', format(v))
        cost += lambda_add * torch.abs(v)
        
        if verbose:
            print('Cost', format(cost))
            
        return cost

    
    B = groups['both']
    M = groups['just_class']
    S = groups['just_confounded']
    N = groups['neither']

    # for compatibility with the SPIRE Code:
    sizes = {'B': B, 'M': M, 'S': S, 'N': N}
    
    # normalize the group counts
    total = B+M+S+N
    B /= total
    M /= total
    S /= total
    N /= total

    # Step 1:  Find how many images to add to each split
    
    deltas = {}
    for c in CHARS:
        deltas[c] = 0.0

    # Sets P(Spurious|Main) = 0.5
    if M > B:
        diff = M - B
        deltas['B'] = diff
    else:
        diff = B - M
        deltas['M'] = diff
        
    # Restore P(Main) while setting P(Spurious|not Main) = 0.5
    P_m = B + M
    P_not_m = S + N
    target = P_not_m * (P_m + diff) / P_m
    if N > target / 2:
        deltas['S'] = target - S - N
    elif S > target / 2:
        deltas['N'] = target - S - N
    else:
        deltas['S'] = target/ 2 - S
        deltas['N'] = target / 2 - N
        
    # Step 2:  Find a sampling procedure to fulfill that order

    augs = {}
    params = []
    for key in ['B2M', 'B2S', 'M2N', 'S2N', 'N2M', 'N2S', 'M2B', 'S2B']:
        augs[key] = torch.tensor(0.0, requires_grad = True)
        params.append(augs[key])

    opt = Adam(params, lr = 0.001)
    hist = []
    for i in range(2000):
        cost = eval_augs(sizes, deltas, augs)
        hist.append(cost.item())

        opt.zero_grad()
        cost.backward()
        opt.step()

        with torch.no_grad():
            for param in params:
                param.clamp_(min = 0)

    eval_augs(sizes, deltas, augs, verbose = True)
    
    # compute the sampling probabilities for the augmentation
    probs = {}
    for key in augs:
        source, dest = unpack_key(key)
        v = augs[key] / max(sizes[source], 1e-8)
        probs[key] = float(v.data.numpy())

    return probs