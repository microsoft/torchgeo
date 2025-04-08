# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

# Check if contextily is available
try:
    import contextily as ctx

    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    ctx = None  # Will be used in the monkeypatch test

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    EDDMapS,
    IntersectionDataset,
    UnionDataset,
)


class TestEDDMapS:
    @pytest.fixture(scope='class')
    def dataset(self) -> EDDMapS:
        root = os.path.join('tests', 'data', 'eddmaps')
        return EDDMapS(root)

    def test_getitem(self, dataset: EDDMapS) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)

    def test_len(self, dataset: EDDMapS) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: EDDMapS) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EDDMapS) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EDDMapS(tmp_path)

    def test_invalid_query(self, dataset: EDDMapS) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_plot(self, dataset: EDDMapS) -> None:
        """Test that the plot method generates a figure without errors."""
        sample = dataset[dataset.bounds]
        fig = dataset.plot(sample)

        # Basic checks to ensure figure was created correctly
        assert fig is not None
        assert isinstance(fig, Figure)

        # Check if the figure has at least one axis
        assert len(fig.axes) > 0

        # Check if a scatter plot exists in the first axis
        assert any(
            isinstance(child, PathCollection) for child in fig.axes[0].get_children()
        )

        # Check if colorbar was added
        assert any(
            'colorbar' in getattr(ax, 'get_label', lambda: '')() for ax in fig.axes
        )

        # Optional: close figure to avoid warnings about too many open figures
        plt.close(fig)

    def test_plot_empty_dates(self, dataset: EDDMapS) -> None:
        """Test plot method with a sample that has no dates."""
        # Create a sample with empty data
        sample = {'bounds': [], 'crs': dataset.crs}
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_CONTEXTILY, reason='contextily not installed')
    def test_plot_basemap_exception(
        self, dataset: EDDMapS, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that plot handles basemap failures gracefully."""

        def mock_add_basemap(*args: object, **kwargs: object) -> None:
            raise Exception('Mocked basemap failure')

        # Replace the real function with our mock
        monkeypatch.setattr(ctx, 'add_basemap', mock_add_basemap)

        # Get a normal sample and plot it
        sample = dataset[dataset.bounds]
        fig = dataset.plot(sample)

        # Check that the grid was added as fallback by examining axis grid properties
        x_axis = fig.axes[0].xaxis
        y_axis = fig.axes[0].yaxis

        # Check if grid lines are enabled and have the expected style
        assert x_axis.get_gridlines()[0].get_visible() is True
        assert y_axis.get_gridlines()[0].get_linestyle() == '--'

        plt.close(fig)

    def test_plot_with_suptitle(self, dataset: EDDMapS) -> None:
        """Test plot method with a custom suptitle."""
        sample = dataset[dataset.bounds]
        test_title = 'Test Suptitle'
        fig = dataset.plot(sample, suptitle=test_title)

        # Check if the suptitle was set
        assert len(fig.texts) > 0
        assert any(text.get_text() == test_title for text in fig.texts)
        plt.close(fig)

    def test_contextily_import_both_cases(self) -> None:
        """Test both cases of contextily import"""
        # Save original modules state
        orig_modules = dict(sys.modules)

        # Case 1: Test with contextily available
        if 'contextily' in sys.modules:
            import importlib

            import torchgeo.datasets.eddmaps

            importlib.reload(torchgeo.datasets.eddmaps)
            assert torchgeo.datasets.eddmaps.HAS_CONTEXTILY is True

        # Case 2: Test with contextily not available
        # Remove contextily from sys.modules to simulate it not being installed
        sys.modules['contextily'] = None  # type: ignore

        # Force reload of the module to trigger the ImportError case
        import importlib

        import torchgeo.datasets.eddmaps

        importlib.reload(torchgeo.datasets.eddmaps)

        assert torchgeo.datasets.eddmaps.HAS_CONTEXTILY is False

        # Restore original modules
        sys.modules.clear()
        sys.modules.update(orig_modules)

        # Reload one more time to restore the original state
        importlib.reload(torchgeo.datasets.eddmaps)

    def test_plot_both_contextily_cases(self, dataset: EDDMapS) -> None:
        """Test plot with both contextily available and not available"""
        import torchgeo.datasets.eddmaps

        # Get a sample to plot
        sample = dataset[dataset.bounds]

        # Case 1: Test with current state
        original_has_contextily = torchgeo.datasets.eddmaps.HAS_CONTEXTILY
        fig1 = dataset.plot(sample)
        plt.close(fig1)

        # Case 2: Test with the opposite state
        with mock.patch.object(
            torchgeo.datasets.eddmaps, 'HAS_CONTEXTILY', not original_has_contextily
        ):
            fig2 = dataset.plot(sample)
            plt.close(fig2)
