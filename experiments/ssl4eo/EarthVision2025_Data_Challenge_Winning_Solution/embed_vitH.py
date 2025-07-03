#!/usr/bin/env python
# coding: utf-8
import argparse
import open_clip
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import glob
from torchvision import transforms
from torch.utils.data import DataLoader

from challenge_dataset import E2SChallengeDataset, collate_fn
from ssl4eos12_dataset import SSL4EOS12Dataset, S2L1C_MEAN, S2L1C_STD, S2L2A_MEAN, S2L2A_STD, S1GRD_MEAN, S1GRD_STD


def create_submission_from_dict(emb_dict):
    """Assume dictionary has format {hash-id0: embedding0, hash-id1: embedding1, ...}
    """
    df_submission = pd.DataFrame.from_dict(emb_dict, orient='index')

    # Reset index with name 'id'
    df_submission.index.name = 'id'
    df_submission.reset_index(drop=False, inplace=True)
        
    return df_submission

def test_submission(path_to_submission: str, 
                    expected_embedding_ids: set, 
                    embedding_dim: int = 1024):
    # Load data
    df = pd.read_csv(path_to_submission, header=0)

    # Verify that id is in columns
    if 'id' not in df.columns:
        raise ValueError(f"""Submission file must contain column 'id'.""")

    # Temporarily set index to 'id'
    df.set_index('id', inplace=True)

    # Check that all samples are included
    submitted_embeddings = set(df.index.to_list())
    n_missing_embeddings = len(expected_embedding_ids.difference(submitted_embeddings))
    if n_missing_embeddings > 0:
        raise ValueError(f"""Submission is missing {n_missing_embeddings} embeddings.""")
    
    # Check that embeddings have the correct length
    if len(df.columns) != embedding_dim:
        raise ValueError(f"""{embedding_dim} embedding dimensions, but provided embeddings have {len(df.columns)} dimensions.""")

    # Convert columns to float
    try:
        for col in df.columns:
            df[col] = df[col].astype(float)
    except Exception as e:
        raise ValueError(f"""Failed to convert embedding values to float.
    Check embeddings for any not-allowed character, for example empty strings, letters, etc.
    Original error message: {e}""")

    # Check if any NaNs 
    if df.isna().any().any():
        raise ValueError(f"""Embeddings contain NaN values.""")
    
    # Successful completion of the function
    return True


parser = argparse.ArgumentParser(description="Embed the dataset using GeoRSCLIP")

# Required args
parser.add_argument("--dataset-path", "-d", type=str, required=True, help="Path to the root of the Zarr dataset, e.g., /data/embed2scale/data_eval")
parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the GeoRSCLIP checkpoint")
parser.add_argument("--output-path", "-o", type=str, default="embeddings_clip_VITH_256_E25.csv", help="Path to the output CSV file")
args = parser.parse_args()



# Path to challenge data folder, i.e. the folder containing the s1, s2l1c and s2l2a subfolders.
path_to_data = args.dataset_path

# Path to where the submission file should be written.
path_to_output_file = args.output_path 

# Path to the ckpt
ckpt = args.ckpt_path # '/mnt/disk3/SSL4EO_TEST/vith/epoch_25.pt'




modalities = ['s2l1c', 's2l2a', 's1']
mean_data = S2L1C_MEAN + S2L2A_MEAN + S1GRD_MEAN
std_data = S2L1C_STD + S2L2A_STD + S1GRD_STD

data_transform = transforms.Compose([
    # Add additional transformation here
    transforms.Normalize(mean=mean_data, std=std_data)
])

concatenate_modalities = True
dataset_e2s = E2SChallengeDataset(path_to_data, 
                                  modalities = modalities, 
                                  dataset_name='bands', 
                                  transform=data_transform, 
                                  concat=concatenate_modalities,
                                  output_file_name=True,
                                  shift_s2_channels=True
                                  )

# Print dataset length
print(f"Length of dataset: {len(dataset_e2s)}")

# Print shape of first sample
print("Data example shape: ", dataset_e2s[0]['data'].shape)


train_loader  = DataLoader(
    dataset=dataset_e2s,
    batch_size=1,  # Note that each each challenge task zarr file contains a single sample.
    shuffle=False,
    collate_fn=collate_fn,  # Data needs to be concatenated along sample dimension instead of being stacked.
    pin_memory=True, 
    num_workers=8 # Trying to speed up the data loading.
)


print("loading checkpoints...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=ckpt)

# checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
# print(checkpoint['state_dict']['module.positional_embedding'].shape)

model.eval()
model.cuda()
with torch.no_grad():
    embeddings = {}
    for ind, data_file_name in tqdm(enumerate(train_loader), total=len(train_loader)):
        file_names = data_file_name['file_name']
        images = data_file_name['data'].reshape(data_file_name['data'].shape[0], 108, 264, 264)
        images = images.to('cuda', non_blocking=True)
        features = model(preprocess(images))  # Get features (not the projection head output)
        features = features[0].cpu().numpy()
        # Resize feature shape to (, 1024) by concatenating itself multiple times
        full_repeats = 1024 // features.shape[1]
        remainder = 1024 % features.shape[1]
        features_repeated = np.concatenate([features] * full_repeats + [features[:, :remainder]], axis=1)
        # Add each file name and corresponding (1024) vector to the embeddings dictionary
        for i, file_name in enumerate(file_names):
            embeddings[file_name] = features_repeated[i]


submission_file = create_submission_from_dict(embeddings)
print('Number of embeddings:', len(submission_file))

# Set to True to trigger saving of the csv at the end.
write_result_to_file = True

# Write submission
if write_result_to_file:
    submission_file.to_csv(path_to_output_file, index=False)

# We use the created embeddings as the list of all samples.
# This can be done since we are sure to have fully looped through the dataset.
# A better way would be to find all the IDs in the challenge data separately, e.g. from the dataloader.

embedding_dim = 1024
embedding_ids = set(embeddings.keys())

# Test submission
print(test_submission(path_to_output_file, embedding_ids, embedding_dim))



