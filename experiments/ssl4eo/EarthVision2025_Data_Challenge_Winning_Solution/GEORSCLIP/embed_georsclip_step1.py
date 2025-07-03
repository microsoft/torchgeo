import argparse

from PIL import Image
import numpy as np
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as tud

from dataloader import L2AZarrDataset


def main():
    def collate(batch: list[dict[str, np.ndarray]]) -> torch.Tensor:
        """
        Args:
            batch: list[dict[str, np.ndarray[4, ...]]] for four seasons

        Returns:
            torch.Tensor[batch_size * 4, ...]
        """
        out = [eval_transform(Image.fromarray(ex["rgb"][idx])) for ex in batch for idx in range(4)]
        return torch.stack(out, dim=0)

    parser = argparse.ArgumentParser(description="Embed the dataset using GeoRSCLIP")
    
    # Required args
    parser.add_argument("--dataset-path", "-d", type=str, required=True, help="Path to the root of the Zarr dataset, e.g., /data/embed2scale/data_eval")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the GeoRSCLIP checkpoint")
    
    # Optional args
    parser.add_argument("--output-path", "-o", type=str, default="georsclip-embeddings-step1.pt", help="Path to the output embeddings file")
    parser.add_argument("--model-name", "-m", type=str, default="ViT-H-14", help="Name of the model to use; see GeoRSCLIP")
    parser.add_argument("--pretrained-name", "-p", type=str, default="laion2b_s32b_b79k", help="Name of the pretrained arch to use; see GeoRSCLIP")
    parser.add_argument("--device", "-dev", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    # Load the model
    model, _, eval_transform = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained_name,
        device=args.device,
    )

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(args.device)
    visual_model = torch.compile(model.visual)
    torch.set_float32_matmul_precision("high")

    # Load the dataset
    ds = L2AZarrDataset(args.dataset_path)
    dl = tud.DataLoader(ds, batch_size=24, collate_fn=collate, num_workers=8, prefetch_factor=2)

    # Embed the dataset
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Embedding dataset"):
            batch = batch.to(args.device)
            emb = nn.functional.normalize(visual_model(batch), dim=-1)
            embeddings.append(emb.view(batch.size(0) // 4, 4, -1).cpu())

    # Save the embeddings (num_examples, 4, num_features)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    torch.save(embeddings, args.output_path)


if __name__ == "__main__":
    main()