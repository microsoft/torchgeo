1|# Copyright (c) TorchGeo Contributors. All rights reserved.
2|# Licensed under the MIT License.
3|
4|import glob
5|import os
6|import shutil
7|from pathlib import Path
8|
9|import matplotlib.pyplot as plt
10|import pytest
11|import torch
12|from _pytest.fixtures import SubRequest
13|from pytest import MonkeyPatch
14|from torch import nn
15|
16|from torchgeo.datasets import DatasetNotFoundError, IDTReeS
17|
18|pytest.importorskip('laspy', minversion='2.5.3')
19|
20|
21|class TestIDTReeS:
22|    @pytest.fixture(params=zip(['train', 'test', 'test'], ['task1', 'task1', 'task2']))
23|    def dataset(
24|        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
25|    ) -> IDTReeS:
26|        data_dir = os.path.join('tests', 'data', 'idtrees')
27|        metadata = {
28|            'train': {
29|                'url': os.path.join(data_dir, 'IDTREES_competition_train_v2.zip'),
30|                'md5': '',
31|                'filename': 'IDTREES_competition_train_v2.zip',
32|            },
33|            'test': {
34|                'url': os.path.join(data_dir, 'IDTREES_competition_test_v2.zip'),
35|                'md5': '',
36|                'filename': 'IDTREES_competition_test_v2.zip',
37|            },
38|        }
39|        split, task = request.param
40|        monkeypatch.setattr(IDTReeS, 'metadata', metadata)
41|        root = tmp_path
42|        transforms = nn.Identity()
43|        return IDTReeS(root, split, task, transforms, download=True)
44|
45|    def test_getitem(self, dataset: IDTReeS) -> None:
46|        x = dataset[0]
47|        assert isinstance(x, dict)
48|        assert isinstance(x['image'], torch.Tensor)
49|        assert isinstance(x['chm'], torch.Tensor)
50|        assert isinstance(x['hsi'], torch.Tensor)
51|        assert isinstance(x['las'], torch.Tensor)
52|        assert x['image'].shape == (3, 200, 200)
53|        assert x['chm'].shape == (1, 200, 200)
54|        assert x['hsi'].shape == (369, 200, 200)
55|        assert x['las'].ndim == 2
56|        assert x['las'].shape[0] == 3
57|
58|        if 'label' in x:
59|            assert isinstance(x['label'], torch.Tensor)
60|        if 'bbox_xyxy' in x:
61|            assert isinstance(x['bbox_xyxy'], torch.Tensor)
62|            if x['bbox_xyxy'].ndim != 1:
63|                assert x['bbox_xyxy'].ndim == 2
64|                assert x['bbox_xyxy'].shape[-1] == 4
65|                assert x['bbox_xyxy'].shape[0] > 0
66|
67|    def test_len(self, dataset: IDTReeS) -> None:
68|        assert len(dataset) == 3
69|
70|    def test_already_downloaded(self, dataset: IDTReeS) -> None:
71|        IDTReeS(root=dataset.root, download=True)
72|
73|    def test_not_downloaded(self, tmp_path: Path) -> None:
74|        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
75|            IDTReeS(tmp_path)
76|
77|    def test_not_extracted(self, tmp_path: Path) -> None:
78|        pathname = os.path.join('tests', 'data', 'idtrees', '*.zip')
79|        root = tmp_path
80|        for zipfile in glob.iglob(pathname):
81|            shutil.copy(zipfile, root)
82|        IDTReeS(root)
83|
84|    def test_plot(self, dataset: IDTReeS) -> None:
85|        x = dataset[0].copy()
86|        dataset.plot(x, suptitle='Test')
87|        plt.close()
88|        dataset.plot(x, show_titles=False)
89|        plt.close()
90|
91|        if 'bbox_xyxy' in x:
92|            x['prediction_bbox_xyxy'] = x['bbox_xyxy']
93|            dataset.plot(x, show_titles=True)
94|            plt.close()
95|        if 'label' in x:
96|            x['prediction_label'] = x['label']
97|            dataset.plot(x, show_titles=False)
98|            plt.close()
99|
100|    def test_empty_boxes(self, dataset: IDTReeS, monkeypatch: MonkeyPatch) -> None:
101|        def mock_load_boxes(self, path: Path) -> list:
102|            return []
103|        monkeypatch.setattr(IDTReeS, '_load_boxes', mock_load_boxes)
104|        x = dataset[0]
105|        assert 'bbox_xyxy' in x
106|        assert x['bbox_xyxy'].shape == (0, 4)
