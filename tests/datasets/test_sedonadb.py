import os
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from pyproj import CRS

from torchgeo.datasets import DatasetNotFoundError, SedonaDBDataset

pytest.importorskip('sedonadb')


class CustomVectorDataset(SedonaDBDataset):
    filename_glob = '*.geojson'
    date_format = '%Y'
    filename_regex = r"""
        ^vector_(?P<date>\d{4})\.geojson
    """


class CustomVectorParquetDataset(SedonaDBDataset):
    filename_glob = '*.parquet'
    date_format = '%Y'
    filename_regex = r"""
        ^vector_(?P<date>\d{4})\.parquet
    """


class TestSedonaDBDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomVectorDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorDataset(root, res=(0.1, 0.1), transforms=transforms)

    @pytest.fixture(scope='class')
    def multilabel(self) -> CustomVectorDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorDataset(
            root, res=(0.1, 0.1), transforms=transforms, label_name='label_id'
        )

    @pytest.fixture(scope='class')
    def dataset_parquet(self) -> CustomVectorParquetDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorParquetDataset(root, res=(0.1, 0.1), transforms=transforms)

    def test_invalid_task(self, dataset: CustomVectorDataset) -> None:
        with pytest.raises(ValueError, match='Invalid task:'):
            CustomVectorDataset(dataset.paths, task='invalid-task')

    def test_getitem(self, dataset: CustomVectorDataset) -> None:
        dataset.task = 'semantic_segmentation'
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(x['mask'].unique(), torch.tensor([0, 1], dtype=torch.uint8))

        dataset.task = 'object_detection'
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['bbox_xyxy'].shape[-1] == 4

        dataset.task = 'instance_segmentation'
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(x['mask'].unique(), torch.tensor([0, 1], dtype=torch.uint8))
        assert x['bbox_xyxy'].shape[-1] == 4
        assert len(x['label']) == x['mask'].shape[0]

    def test_time_index(self, dataset: CustomVectorDataset) -> None:
        assert dataset.bounds[2].start > pd.Timestamp.min
        assert dataset.bounds[2].stop < pd.Timestamp.max

    def test_getitem_multilabel(self, multilabel: CustomVectorDataset) -> None:
        multilabel.task = 'semantic_segmentation'
        x = multilabel[multilabel.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(
            x['mask'].unique(), torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
        )

        multilabel.task = 'object_detection'
        x = multilabel[multilabel.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert torch.equal(x['label'], torch.tensor([1, 2, 3], dtype=torch.int32))
        assert x['bbox_xyxy'].shape[-1] == 4

        multilabel.task = 'instance_segmentation'
        x = multilabel[multilabel.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert torch.equal(x['label'], torch.tensor([1, 2, 3], dtype=torch.int32))
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(x['mask'].unique(), torch.tensor([0, 1], dtype=torch.uint8))
        assert x['bbox_xyxy'].shape[-1] == 4
        assert len(x['label']) == x['mask'].shape[0]

    def test_getitem_parquet(self, dataset_parquet: CustomVectorParquetDataset) -> None:
        dataset_parquet.task = 'semantic_segmentation'
        x = dataset_parquet[dataset_parquet.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(x['mask'].unique(), torch.tensor([0, 1], dtype=torch.uint8))

    def test_getitem_with_layer(self) -> None:
        root = os.path.join('tests', 'data', 'vector')
        dataset = CustomVectorDataset(
            root, res=(0.1, 0.1), transforms=nn.Identity(), layer=0
        )
        dataset.task = 'semantic_segmentation'
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_empty_shapes(self, dataset: CustomVectorDataset) -> None:
        dataset.task = 'semantic_segmentation'
        x = dataset[1.1:1.9, 1.1:1.9, pd.Timestamp.min : pd.Timestamp.max]
        assert torch.equal(x['mask'], torch.zeros(8, 8, dtype=torch.uint8))

        dataset.task = 'object_detection'
        x = dataset[1.1:1.9, 1.1:1.9, pd.Timestamp.min : pd.Timestamp.max]
        assert torch.equal(x['bbox_xyxy'], torch.empty(0, 4, dtype=torch.float32))

        dataset.task = 'instance_segmentation'
        x = dataset[1.1:1.9, 1.1:1.9, pd.Timestamp.min : pd.Timestamp.max]
        assert torch.equal(x['bbox_xyxy'], torch.empty(0, 4, dtype=torch.float32))
        assert torch.equal(x['mask'], torch.zeros(8, 8, dtype=torch.uint8))

    def test_invalid_query(self, dataset: CustomVectorDataset) -> None:
        with pytest.raises(
            IndexError, match=r'query: .* not found in index with bounds:'
        ):
            dataset[3:3, 3:3, pd.Timestamp.min : pd.Timestamp.min]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SedonaDBDataset(tmp_path)

    def test_single_res(self) -> None:
        root = os.path.join('tests', 'data', 'vector')
        ds = CustomVectorDataset(root, res=0.1)
        assert ds.res == (0.1, 0.1)

    def test_skip_unreadable_file(self, tmp_path: Path) -> None:
        valid_file = tmp_path / 'vector_2024.geojson'
        invalid_file = tmp_path / 'vector_2025.geojson'
        valid_file.write_text(
            '{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]}}]}'
        )
        invalid_file.write_text('invalid geojson content')

        ds = CustomVectorDataset(tmp_path, res=(0.1, 0.1))
        assert len(ds) == 1
        assert str(valid_file) in [str(fp) for fp in ds.index['filepath']]
        assert str(invalid_file) not in [str(fp) for fp in ds.index['filepath']]
