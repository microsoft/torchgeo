# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DetCon-B: DetCon implementation for BYOL."""
from torch.nn.modules import Module
import torch.nn.functional as F
import numpy as np
from torchgeo.Trainer import byol
import skimage

Module.__module__ = "torch.nn"


def featurewise_std(x: np.ndarray) -> np.ndarray:
    """Computes the featurewise standard deviation."""
    return np.mean(np.std(x, axis=0))


def compute_fh_segmentation(image_np, scale, min_size):
  """Compute FSZ segmentation on image and record stats."""
    segmented_image = skimage.segmentation.felzenszwalb(
        image_np, scale=scale, min_size=min_size)
    segmented_image = segmented_image.astype(np.dtype('<u1'))
    return segmented_image


class DetConB(Module):
    """DetCon-B's training component definition."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        assert self.hparams["training_mode"] in ['self-supervised', 'supervised', 'both']
        self.training_mode = self.hparams["training_mode"]


    def __init__(self, mode: Text, model: Module,
        image_size: Tuple[int, int] = (224, 224),
        hidden_layer: Union[str, int] = -2,
        input_channels: int = 3,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Optional[Module] = None,
        beta: float = 0.99, **kwargs: Any):
        """Constructs the experiment.
        Args:
        mode: A string, equivalent to FLAGS.mode when running normally.
        config: Experiment configuration.
        """
        super().__init__(mode, kwargs)

        self.augment: Module
        if augment_fn is None:
            self.augment = byol.SimCLRAugmentation(image_size)
        else:
            self.augment = augment_fn

        self.beta = beta
        self.input_channels = input_channels
        self.encoder = byol.EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = byol.MLP(projection_size, projection_size, hidden_size)
        self._target: Optional[Module] = None

        # Perform a single forward pass to initialize the wrapper correctly
        self.encoder(
            torch.zeros(  # type: ignore[attr-defined]
                2, self.input_channels, *image_size
            )
        )

        self.config_task()

    def create_binary_mask(
        self,
        batch_size,
        num_pixels,
        masks,
        max_mask_id=256,
        downsample=(1, 32, 32, 1)):
        """Generates binary masks from the Felzenszwalb masks.
        From a FH mask of shape [batch_size, H,W] (values in range
        [0,max_mask_id], produces corresponding (downsampled) binary masks of
        shape [batch_size, max_mask_id, H*W/downsample].
        Args:
            batch_size: batch size of the masks
            num_pixels: Number of points on the spatial grid
            masks: Felzenszwalb masks
            max_mask_id: # unique masks in Felzenszwalb segmentation
            downsample: rate at which masks must be downsampled.
        Returns:
            binary_mask: Binary mask with specification above
        """
        fh_mask_to_use = self.hparams["fh_mask_to_use"]
        mask = masks[..., fh_mask_to_use:(fh_mask_to_use+1)]

        mask_ids = np.arange(max_mask_id).reshape(1, 1, 1, max_mask_id)
        binary_mask = np.equal(mask_ids, mask).astype('float32')

        binary_mask = F.avg_pool2d(binary_mask, downsample, downsample, count_include_pad=False)
        binary_mask = binary_mask.reshape(batch_size, num_pixels, max_mask_id)
        binary_mask = np.argmax(binary_mask, axis=-1)
        binary_mask = np.eye(max_mask_id)[binary_mask]
        binary_mask = np.transpose(binary_mask, [0, 2, 1])
        return binary_mask

    def sample_masks(self, binary_mask, batch_size, n_random_vectors=16):
        """Samples which binary masks to use in the loss."""
        mask_exists = np.greater(binary_mask.sum(-1), 1e-3)
        sel_masks = mask_exists.astype('float32') + 0.00000000001
        sel_masks = sel_masks / sel_masks.sum(1, keepdims=True)
        sel_masks = np.log(sel_masks)

        mask_ids = np.random.choice(
            np.arange(len(sel_masks[-1])), p=sel_masks,
            shape=tuple([n_random_vectors, batch_size]))
        mask_ids = np.transpose(mask_ids, [1, 0])

        smpl_masks = np.stack(
            [binary_mask[b][mask_ids[b]] for b in range(batch_size)])
        return smpl_masks, mask_ids

    @property
    def target(self) -> Module:
        """The "target" model."""
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self) -> None:
        """Method to update the "target" model weights."""
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the encoder model through the MLP and prediction head.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.predictor(self.encoder(x)))

    
    def run_detcon_b_forward_on_view(
        view_encoder: Any,
        projector: Any,
        predictor: Any,
        classifier: Any,
        is_training: bool,
        images: np.ndarray,
        masks: np.ndarray,
        suffix: Text = '',
       
        ):
        pass


    def _forward(
      self,
      inputs: image_dataset.Batch,
      is_training: bool,
  ) -> Mapping[Text, np.ndarray]:
    """Forward application of byol's architecture.
    Args:
      inputs: A batch of data, i.e. a dictionary, with either two keys,
        (`images` and `labels`) or three keys (`view1`, `view2`, `labels`).
      is_training: Training or evaluating the model? When True, inputs must
        contain keys `view1` and `view2`. When False, inputs must contain key
        `images`.
    Returns:
      All outputs of the model, i.e. a dictionary with projection, prediction
      and logits keys, for either the two views, or the image.
    """
    pass
