# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Forest Change dataset."""

import json
import os
from collections.abc import Callable
from random import randint
from typing import Any, ClassVar, Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_and_extract_archive


class ForestChange(NonGeoDataset):
    """Forest change detection and captioning dataset.

    The `Forest-Change
    <https://huggingface.co/datasets/JimmyBrocko/Forest-Change>`__
    dataset is the first benchmark designed for joint forest change detection
    and captioning in remote sensing imagery.  It provides bi-temporal
    satellite image pairs from Google Earth (Landsat), pixel-level deforestation masks, and
    multi-granularity natural-language captions describing forest cover
    changes in tropical and subtropical regions.

    Captions are a fundamental part of the dataset and are always loaded;
    whether they are consumed by a model is a downstream concern.

    Dataset features:

    * 334 annotated bi-temporal RGB image pairs at ~30 m/pixel resolution
    * binary change masks (no change = 0, deforestation = 1)
    * five natural-language captions per image pair describing the change
    * geographic focus on tropical and subtropical deforestation fronts

    Dataset format:

    * images are three-channel PNGs under
      ``<root>/Forest-Change-dataset/images/<split>/A/`` and ``B/``
    * masks are single-channel PNGs under
      ``<root>/Forest-Change-dataset/images/<split>/label/``
    * raw captions are stored in
      ``<root>/Forest-Change-dataset/ForestChatcaptions.json``
    * on first use the dataset preprocesses captions into
      ``<root>/Forest-Change-dataset/tokens/``, ``vocab.json``, and
      per-split list files; subsequent loads skip preprocessing

    Dataset classes:

    0. no change
    1. deforestation

    If you use this dataset in your research, please cite:

    * https://www.sciencedirect.com/science/article/pii/S1574954126001470

    .. versionadded:: 0.10
    """

    splits = ('train', 'val', 'test')

    classes = ('no_change', 'deforestation')

    directories = ('A', 'B', 'label')

    directory = 'Forest-Change-dataset'

    token_directory = 'tokens'

    vocab_filename = 'vocab'

    captions_filename = 'ForestChatcaptions.json'

    special_tokens: ClassVar[dict[str, int]] = {
        '<NULL>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
    }

    url = 'https://hf.co/datasets/JimmyBrocko/Forest-Change/resolve/e8b25bf09c85ec85633d1b1b554f7bb23e47724d/Forest-Change-dataset.zip'
    sha256 = '424931a075f00f8cf21d4d2f622df688de559494844df4876b59bde13d3d855d'
    filename = 'Forest-Change-dataset.zip'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
        max_length: int = 42,
        allow_unknown: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialise a new dataset instance.

        Args:
            root: root directory where the dataset can be found or will be
                downloaded.  The zip extracts to
                ``<root>/Forest-Change-dataset/``.
            split: one of ``'train'``, ``'val'``, or ``'test'``.
            transforms: optional callable applied to each sample dict.
                Should include any normalisation; if ``None`` a built-in
                mean/std normalisation is applied.
            max_length: maximum token sequence length used when encoding
                captions.
            allow_unknown: whether unknown tokens are mapped to ``<UNK>``
                during encoding.  When ``False`` a ``KeyError`` is raised
                for out-of-vocabulary tokens.
            download: if ``True``, download the dataset if it is not
                already present under ``root``.
            checksum: if ``True``, verify the SHA256 of the downloaded zip
                (may be slow).

        Raises:
            AssertionError: if ``split`` is not a valid split name.
            DatasetNotFoundError: if the dataset is not found and
                ``download`` is ``False``.
        """
        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.max_length = max_length
        self.allow_unknown = allow_unknown
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        if not self._check_preprocessed():
            self._preprocess()

        vocab_path = os.path.join(
            str(root), self.directory, self.vocab_filename + '.json'
        )
        with open(vocab_path) as f:
            self.word_vocab: dict[str, int] = json.load(f)

        self.idx_to_word = {v: k for k, v in self.word_vocab.items()}

        self.files = self._load_files()

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at *index*.

        Every sample dict contains:

        * ``'image'``         - float32 ``(2, C, H, W)`` stacked bi-temporal
          tensor
        * ``'mask'``          - int64 ``(1, H, W)`` binary change mask
        * ``'name'``          - filename stem (str)
        * ``'token_all'``     - int64 ``(N, max_length)`` all encoded captions
        * ``'token_all_len'`` - int64 ``(N, 1)`` length of each caption
        * ``'token'``         - int64 ``(max_length,)`` selected caption
        * ``'token_len'``     - scalar int64 length of selected caption

        For train split entries with a caption-index suffix in the split
        file (e.g. ``train_000037.png-0``), the indexed caption is returned
        as ``token`` / ``token_len``.  For all other entries a caption is
        chosen at random.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        """
        f = self.files[index]
        image1 = self._load_image(f['image1'])
        image2 = self._load_image(f['image2'])
        mask = self._load_target(f['mask'])

        stem = f['name'].split('_aug')[0].split('_rep')[0]
        token_path = os.path.join(
            str(self.root), self.directory, self.token_directory, stem + '.txt'
        )

        sample: Sample = {
            'image': torch.stack([image1, image2]),
            'mask': mask,
            'name': f['name'],
            **self._load_tokens(token_path, f['token_id']),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each
                panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 3
        if 'prediction' in sample:
            ncols += 1

        image1 = sample['image'][0].permute(1, 2, 0).numpy().astype(np.uint8)

        image2 = sample['image'][1].permute(1, 2, 0).numpy().astype(np.uint8)

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10))

        axs[0].imshow(image1)
        axs[0].axis('off')

        axs[1].imshow(image2)
        axs[1].axis('off')

        axs[2].imshow(sample['mask'][0], cmap='gray', interpolation='none')
        axs[2].axis('off')

        if 'prediction' in sample:
            axs[3].imshow(sample['prediction'][0], cmap='gray', interpolation='none')
            axs[3].axis('off')

            if show_titles:
                axs[3].set_title('Prediction')

        if show_titles:
            axs[0].set_title('Image 1')
            axs[1].set_title('Image 2')
            axs[2].set_title('Mask')

        captions = [self._decode_tokens(tokens) for tokens in sample['token_all']]

        caption_text = '\n'.join(
            f'{i + 1}. {caption}' for i, caption in enumerate(captions)
        )

        fig.text(
            0.5,
            0.01,
            f'Captions:\n{caption_text}',
            ha='center',
            va='bottom',
            wrap=True,
            fontsize=10,
        )

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout(rect=(0, 0.02, 1, 1))

        return fig

    def _check_integrity(self) -> bool:
        """Check that the raw extracted dataset exists under ``root``.

        Returns:
            ``True`` if the image directories and captions JSON are present
        """
        captions_path = os.path.join(
            str(self.root), self.directory, self.captions_filename
        )
        if not os.path.exists(captions_path):
            return False
        for split in self.splits:
            for directory in self.directories:
                if not os.path.exists(
                    os.path.join(
                        str(self.root), self.directory, 'images', split, directory
                    )
                ):
                    return False
        return True

    def _check_preprocessed(self) -> bool:
        """Check that preprocessed tokens, vocab, and split files exist.

        Returns:
            ``True`` if all preprocessing outputs are present
        """
        base = os.path.join(str(self.root), self.directory)
        if not os.path.exists(os.path.join(base, self.vocab_filename + '.json')):
            return False
        if not os.path.exists(os.path.join(base, self.token_directory)):
            return False
        for split in self.splits:
            if not os.path.exists(os.path.join(base, f'{split}.txt')):
                return False
        return True

    def _download(self) -> None:
        """Download and extract the dataset zip."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.sha256 if self.checksum else None,
        )

    def _preprocess(self) -> None:
        """Preprocess raw captions JSON into per-image token files and vocab.

        Reads ``ForestChatcaptions.json``, tokenises each caption, writes
        individual ``tokens/<stem>.txt`` files, ``vocab.json``, and the
        per-split list ``.txt`` files.  Runs once on first use; subsequent
        instantiations skip this step.
        """
        print('Preprocessing captions (one-time operation)...')
        base = os.path.join(str(self.root), self.directory)
        token_dir = os.path.join(base, self.token_directory)
        os.makedirs(token_dir, exist_ok=True)

        captions_path = os.path.join(base, self.captions_filename)
        with open(captions_path) as f:
            data: dict[str, Any] = json.load(f)

        all_cap_tokens: list[tuple[str, list[list[str]]]] = []
        for img in data['images']:
            tokens_list: list[list[str]] = []
            for sentence in img['sentences']:
                if not sentence['raw']:
                    continue
                tokens_list.append(
                    self._tokenize(
                        sentence['raw'],
                        add_start_token=True,
                        add_end_token=True,
                        punct_to_keep=[';', ','],
                        punct_to_remove=['?', '.'],
                    )
                )
            all_cap_tokens.append((img['filename'], tokens_list))

        all_cap_tokens.sort()

        for split in self.splits:
            list_path = os.path.join(base, f'{split}.txt')
            if os.path.exists(list_path):
                os.remove(list_path)

        for filename, tokens_list in all_cap_tokens:
            stem = os.path.splitext(filename)[0]
            with open(os.path.join(token_dir, stem + '.txt'), 'w') as f:
                json.dump(tokens_list, f)

            prefix = stem.split('_')[0]
            if prefix in self.splits:
                with open(os.path.join(base, f'{prefix}.txt'), 'a') as f:
                    f.write(filename + '\n')

        vocab = self._build_vocab(all_cap_tokens)
        with open(os.path.join(base, self.vocab_filename + '.json'), 'w') as f:
            json.dump(vocab, f)

    @staticmethod
    def _tokenize(
        s: str,
        delim: str = ' ',
        add_start_token: bool = True,
        add_end_token: bool = True,
        punct_to_keep: list[str] | None = None,
        punct_to_remove: list[str] | None = None,
    ) -> list[str]:
        """Tokenise a string into a list of lowercase string tokens.

        Numbers are preserved verbatim.  Punctuation is either kept as
        separate tokens, removed, or left in place according to the
        ``punct_to_keep`` and ``punct_to_remove`` arguments.

        Args:
            s: input sentence
            delim: token delimiter
            add_start_token: prepend ``<START>``
            add_end_token: append ``<END>``
            punct_to_keep: punctuation marks to retain as separate tokens
            punct_to_remove: punctuation marks to strip entirely

        Returns:
            list of string tokens
        """
        s = s.lower()
        parts: list[str] = []
        for word in s.split():
            if word.replace('.', '', 1).isdigit():
                parts.append(word)
            else:
                if punct_to_keep:
                    for p in punct_to_keep:
                        word = word.replace(p, f'{delim}{p}{delim}')
                if punct_to_remove:
                    for p in punct_to_remove:
                        word = word.replace(p, '')
                parts.append(word)

        tokens = [t for t in ' '.join(parts).split(delim) if t]
        if add_start_token:
            tokens.insert(0, '<START>')
        if add_end_token:
            tokens.append('<END>')
        return tokens

    def _build_vocab(
        self, sequences: list[tuple[str, list[list[str]]]], min_token_count: int = 5
    ) -> dict[str, int]:
        """Build a token-to-index vocabulary from tokenised captions.

        Args:
            sequences: list of ``(filename, list_of_token_lists)`` pairs
            min_token_count: minimum frequency for a token to be included

        Returns:
            dict mapping token strings to integer indices
        """
        token_to_count: dict[str, int] = {}
        for _, token_lists in sequences:
            for token_list in token_lists:
                for token in token_list:
                    token_to_count[token] = token_to_count.get(token, 0) + 1

        token_to_idx: dict[str, int] = dict(self.special_tokens)
        for token, count in sorted(token_to_count.items()):
            if token in token_to_idx:
                continue
            if count >= min_token_count:
                token_to_idx[token] = len(token_to_idx)
        return token_to_idx

    def _encode(self, seq_tokens: list[str], token_to_idx: dict[str, int]) -> list[int]:
        """Encode a list of string tokens into integer indices.

        Args:
            seq_tokens: list of string tokens
            token_to_idx: vocabulary mapping

        Returns:
            list of integer indices

        Raises:
            KeyError: if a token is not in the vocabulary and
                ``allow_unknown`` is ``False``
        """
        seq_idx: list[int] = []
        for token in seq_tokens:
            if token not in token_to_idx:
                if self.allow_unknown:
                    token = '<UNK>'
                else:
                    raise KeyError(f'Token "{token}" not in vocab')
            seq_idx.append(token_to_idx[token])
        return seq_idx

    def _decode_tokens(self, tokens: Tensor) -> str:
        """Convert an encoded caption tensor into a human-readable sentence.

        Decodes integer token indices using the dataset vocabulary, removes
        special control tokens such as ``<START>`` and ``<NULL>``, and stops
        decoding at the first ``<END>`` token.

        Args:
            tokens: Tensor of token indices representing an encoded caption.

        Returns:
            Decoded caption string.
        """
        words = []

        for idx in tokens.tolist():
            word = self.idx_to_word.get(idx, '<UNK>')

            if word == '<END>':
                break

            if word not in {'<START>', '<NULL>'}:
                words.append(word)

        return ' '.join(words)

    def _load_files(self) -> list[dict[str, Any]]:
        """Build the file list from the split text file.

        Returns:
            list of dicts with keys ``image1``, ``image2``, ``mask``,
            ``token_id``, and ``name``
        """
        base = os.path.join(str(self.root), self.directory)
        list_path = os.path.join(base, f'{self.split}.txt')
        with open(list_path) as f:
            img_ids = [line.strip() for line in f if line.strip()]

        files: list[dict[str, Any]] = []
        for name in img_ids:
            if self.split == 'train' and '-' in name:
                base_name = name.split('-')[0]
                token_id: int | None = int(name.split('-')[-1])
            else:
                base_name = name
                token_id = None

            stem = os.path.splitext(base_name)[0]
            img_dir = os.path.join(base, 'images', self.split)
            files.append(
                {
                    'image1': os.path.join(img_dir, 'A', base_name),
                    'image2': os.path.join(img_dir, 'B', base_name),
                    'mask': os.path.join(img_dir, 'label', base_name),
                    'token_id': token_id,
                    'name': stem,
                }
            )
        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single RGB image as a float32 CHW tensor.

        Args:
            path: path to the image file

        Returns:
            float32 tensor of shape ``(C, H, W)``
        """
        with Image.open(str(path)) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('RGB'))
            tensor = torch.from_numpy(array).float()
            return einops.rearrange(tensor, 'h w c -> c h w')

    def _load_target(self, path: Path) -> Tensor:
        """Load and binarise a change mask.

        Any non-zero pixel is mapped to class ``1``.

        Args:
            path: path to the mask image

        Returns:
            int64 tensor of shape ``(1, H, W)``
        """
        with Image.open(str(path)) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('L'))
            tensor = torch.from_numpy(array)
            tensor = torch.clamp(tensor, min=0, max=1).to(torch.long)
            return tensor.unsqueeze(0)

    def _load_tokens(self, token_path: Path, token_id: int | None) -> dict[str, Tensor]:
        """Load and encode captions for a single sample.

        Args:
            token_path: path to the JSON caption file
            token_id: index of the caption to use as the primary
                ``token`` / ``token_len`` pair.  When ``None`` a caption
                is chosen at random.

        Returns:
            dict with keys ``token_all``, ``token_all_len``, ``token``,
            ``token_len``
        """
        with open(str(token_path)) as f:
            caption_list: list[list[str]] = json.load(f)

        n = len(caption_list)
        token_all = np.zeros((n, self.max_length), dtype=np.int64)
        token_all_len = np.zeros((n, 1), dtype=np.int64)

        for j, tokens in enumerate(caption_list):
            encoded = self._encode(tokens, self.word_vocab)
            token_all[j, : len(encoded)] = encoded
            token_all_len[j] = len(encoded)

        if token_id is not None:
            token = token_all[token_id]
            token_len = int(token_all_len[token_id, 0])
        else:
            j = randint(0, n - 1)
            token = token_all[j]
            token_len = int(token_all_len[j, 0])

        return {
            'token_all': torch.from_numpy(token_all),
            'token_all_len': torch.from_numpy(token_all_len),
            'token': torch.from_numpy(token.copy()),
            'token_len': torch.tensor(token_len, dtype=torch.int64),
        }
