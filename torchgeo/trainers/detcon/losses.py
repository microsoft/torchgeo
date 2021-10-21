# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loss functions for detcon B."""
import numpy as np
import torch.nn.functional as F
from torchvision.trainers.detcon import detcon_utils


def manual_cross_entropy(labels, logits, weight):
    """Manually computes crossentropy loss.

    Args:
        labels: tensor labels
        logits: tensor logits

    Returns:
        crossentropy loss
    """
  ce = - weight * np.sum(labels * F.log_softmax(logits), axis=-1)
  return np.mean(ce)


def l2_normalize( x: np.ndarray,
                axis: Optional[int] = None,
                epsilon: float = 1e-12,
                ) -> np.ndarray:
    """l2 normalize a tensor on an axis with numerical stability."""
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_inv_norm = 1/(np.sqrt(np.maximum(square_sum, epsilon)))
    return x * x_inv_norm


def byol_nce_detcon(pred1, pred2, target1, target2,
                    pind1, pind2, tind1, tind2,
                    temperature=0.1, use_replicator_loss=True,
                    local_negatives=True):
    """Compute the NCE scores from pairs of predictions and targets.
    This implements the batched form of the loss described in
    Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.
    Args:
        pred1 (np.array): the prediction from first view.
        pred2 (np.array): the prediction from second view.
        target1 (np.array): the projection from first view.
        target2 (np.array): the projection from second view.
        pind1 (np.array): mask indices for first view's prediction.
        pind2 (np.array): mask indices for second view's prediction.
        tind1 (np.array): mask indices for first view's projection.
        tind2 (np.array): mask indices for second view's projection.
        temperature (float): the temperature to use for the NCE loss.
        use_replicator_loss (bool): use cross-replica samples.
        local_negatives (bool): whether to include local negatives
    Returns:
        A single scalar loss for the XT-NCE objective.
    """
    batch_size = pred1.shape[0]
    num_rois = pred1.shape[1]
    feature_dim = pred1.shape[-1]
    infinity_proxy = 1e9  # Used for masks to proxy a very large number.

    def make_same_obj(ind_0, ind_1):
        same_obj = np.equal(ind_0.reshape([batch_size, num_rois, 1]),
                            ind_1.reshape([batch_size, 1, num_rois]))
        return np.expand_dims(same_obj.astype("float32"), axis=2)
    same_obj_aa = make_same_obj(pind1, tind1)
    same_obj_ab = make_same_obj(pind1, tind2)
    same_obj_ba = make_same_obj(pind2, tind1)
    same_obj_bb = make_same_obj(pind2, tind2)

    # L2 normalize the tensors to use for the cosine-similarity
    pred1 = l2_normalize(pred1, axis=-1)
    pred2 = l2_normalize(pred2, axis=-1)
    target1 = l2_normalize(target1, axis=-1)
    target2 = l2_normalize(target2, axis=-1)

    #Just work for a sungle GPU for now

    target1_large = target1
    target2_large = target2
    labels_local = F.one_hot(np.arange(batch_size), batch_size)
    labels_ext = F.one_hot(np.arange(batch_size), batch_size * 2)

    labels_local = np.expand_dims(np.expand_dims(labels_local, axis=2), axis=1)
    labels_ext = np.expand_dims(np.expand_dims(labels_ext, axis=2), axis=1)

    # Do our matmuls and mask out appropriately.
    logits_aa = np.einsum("abk,uvk->abuv", pred1, target1_large) / temperature
    logits_bb = np.einsum("abk,uvk->abuv", pred2, target2_large) / temperature
    logits_ab = np.einsum("abk,uvk->abuv", pred1, target2_large) / temperature
    logits_ba = np.einsum("abk,uvk->abuv", pred2, target1_large) / temperature

    labels_aa = labels_local * same_obj_aa
    labels_ab = labels_local * same_obj_ab
    labels_ba = labels_local * same_obj_ba
    labels_bb = labels_local * same_obj_bb

    logits_aa = logits_aa - infinity_proxy * labels_local * same_obj_aa
    logits_bb = logits_bb - infinity_proxy * labels_local * same_obj_bb
    labels_aa = 0. * labels_aa
    labels_bb = 0. * labels_bb
    if not local_negatives:
        logits_aa = logits_aa - infinity_proxy * labels_local * (1 - same_obj_aa)
        logits_ab = logits_ab - infinity_proxy * labels_local * (1 - same_obj_ab)
        logits_ba = logits_ba - infinity_proxy * labels_local * (1 - same_obj_ba)
        logits_bb = logits_bb - infinity_proxy * labels_local * (1 - same_obj_bb)

    labels_abaa = np.concatenate([labels_ab, labels_aa], axis=2)
    labels_babb = np.concatenate([labels_ba, labels_bb], axis=2)

    labels_0 = np.reshape(labels_abaa, [batch_size, num_rois, -1])
    labels_1 = np.reshape(labels_babb, [batch_size, num_rois, -1])

    num_positives_0 = np.sum(labels_0, axis=-1, keepdims=True)
    num_positives_1 = np.sum(labels_1, axis=-1, keepdims=True)

    labels_0 = labels_0 / np.maximum(num_positives_0, 1)
    labels_1 = labels_1 / np.maximum(num_positives_1, 1)

    obj_area_0 = np.sum(make_same_obj(pind1, pind1), axis=[2, 3])
    obj_area_1 = np.sum(make_same_obj(pind2, pind2), axis=[2, 3])

    weights_0 = np.greater(num_positives_0[..., 0], 1e-3).astype("float32")
    weights_0 = weights_0 / obj_area_0
    weights_1 = np.greater(num_positives_1[..., 0], 1e-3).astype("float32")
    weights_1 = weights_1 / obj_area_1

    logits_abaa = np.concatenate([logits_ab, logits_aa], axis=2)
    logits_babb = np.concatenate([logits_ba, logits_bb], axis=2)

    logits_abaa = np.reshape(logits_abaa, [batch_size, num_rois, -1])
    logits_babb = np.reshape(logits_babb, [batch_size, num_rois, -1])

    loss_a = manual_cross_entropy(labels_0, logits_abaa, weights_0)
    loss_b = manual_cross_entropy(labels_1, logits_babb, weights_1)
    loss = loss_a + loss_b

    return loss