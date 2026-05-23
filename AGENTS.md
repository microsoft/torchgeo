# TorchGeo LLM Instructions

Datasets, samplers, transforms, and pre-trained models for geospatial data.

## Commands

```bash
# Install
pip install -e ".[all]"

# Lint (run from repo root)
ruff format && ruff check && ty check && prettier --write .

# Test
pytest --cov=torchgeo tests/                                                        # all (skip slow)
pytest --cov=torchgeo.datasets tests/datasets/test_foo.py                          # single file
pytest --cov=torchgeo.datasets tests/datasets/test_foo.py::TestFooDataset::test_getitem  # single method
pytest -m "" --cov=torchgeo tests/                                                  # include slow

# Docs (requires Pandoc + pip install ".[docs]")
cd docs && make clean && make html
```

## Project Structure

```
torchgeo/{datamodules,datasets,losses,models,samplers,trainers,transforms}/
tests/{data/<dataset>/,datasets/,...}
```

## Code Style

### File Header (required)

```python
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
```

### Formatting (Ruff)

- Single quotes, no trailing commas
- Imports: isort order (stdlib → third-party → local)
- Local imports (`from .utils`) in `torchgeo/`; absolute (`from torchgeo`) in `tests/`

### Type Hints (ty)

- All functions require full annotations
- Union: `X | Y` (not `Union[X, Y]`)
- Treat `Path` as `type Path = str | os.PathLike[str]` when imported from `torchgeo.datasets.utils`
- Avoid `Any`; use `Sequence` for parameters, `list`/`tuple` for return values
- Prefer built-in types (`list`, `dict`, `tuple`) over `typing.List`, etc.
- `type: ignore` only for external lib issues

### Docstrings (Google style)

```python
def func(x: Tensor, weights: Weights | None = None) -> Tensor:
    """Short description.

    Args:
        x: Input tensor.
        weights: Pre-trained weights.

    Returns:
        Output tensor.

    Raises:
        ValueError: If x is empty.

    Warns:
        UserWarning: If weights are outdated.

    """
```

- Include `Args`, `Returns`, `Raises`, `Warns` sections as needed
- Period after first line; blank line after first sentence
- Don't document default values in Args (Sphinx adds them)
- `.. versionadded::` for new features; `.. deprecated::` for deprecations
- Inline comments only for "why", not "what"

### RST Conventions (docs/docstrings)

- Monospace: double backticks ` ``code`` `
- Bold: `**text**`
- Italics: `*text*`
- Single backticks create links only when followed by underscore; otherwise italics
- Blank line required before bulleted lists

### Classes

```python
class Foo(NonGeoDataset):
    """Dataset description. Citation info.

    .. versionadded:: 0.5
    """
    url = 'https://...'
    classes = ['background', 'building']

    def __init__(self, root: Path = 'data', download: bool = False) -> None:
```

## Testing

```python
class TestFooDataset:
    @pytest.fixture
    def dataset(self) -> FooDataset:
        return FooDataset(root='tests/data')

    def test_getitem(self, dataset: FooDataset) -> None:
        x = dataset[0]
        assert x['image'].ndim == 3
        assert x['mask'].ndim == 2
```

- 100% coverage required; check with `pytest --cov=torchgeo.datasets tests/datasets/test_foo.py`
- Fake data in `tests/data/<dataset>/` (never real data); `data.py` generates it
- `@pytest.mark.slow` for downloads/network
- `pytest.importorskip('pkg', minversion='X.X')` for optional deps
- `plt.close()` at end of plot tests

## Dataset Implementation

```python
class MyDataset(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^(?P<date>\d{8})_.*\.tif$'
    date_format = '%Y%m%d'
    is_image = True
```

- `root='data'` default; `_verify()` for existence checks
- Sample dict: `{"image": tensor, "mask": tensor}`
- PIL for non-geo images, rasterio for geo images

### New Dataset Checklist

1. `torchgeo/datasets/foo.py` (add license header)
2. Import in `torchgeo/datasets/__init__.py`
3. `tests/data/foo/data.py` (fake data matching real structure)
4. `tests/datasets/test_foo.py` (100% coverage)
5. `docs/api/datasets.rst`
6. `docs/api/datasets/[non_]geo_datasets.csv`

## Error Handling

- Never catch bare `Exception`; be specific
- Utilize torchgeo exceptions from `torchgeo.datasets.errors` if applicable, e.g. `DatasetNotFoundError`, `RGBBandsMissingError`, `DependencyNotFoundError`
- Document in `Raises` section
- Optional deps: lazy import + helpful ImportError (`'scipy required for ...'`)
- Use `None` only for mutable defaults; prefer immutable tuples

## Geospatial

- Check "contains" not just "intersects" for sampling
- Auto-detect CRS when not specified
- Warn if datasets have different CRS/res
- Skip warped bounds computation if CRS matches

## Transforms (Kornia)

- GPU augs in `on_after_batch_transfer`
- Cast mask to float before Kornia, restore dtype after
- Masks need channel dim for Kornia, squeeze before loss
- `AugmentationSequential` requires `data_keys`

## Lightning

- `prepare_data`: downloads only, NO state
- `setup`: create dataset objects
- `on_after_batch_transfer`: GPU augmentations

## Performance

- Class attributes for reusable objects (not per-call instantiation)
- Prefer `einops` operations for readability, like `einops.rearrange` and `einops.reshape`; always specify `dim` in `squeeze()`
- Checksums slow—make optional

## Models

- Document input/output shapes
- When modifying conv: update both weight AND `in_channels`
- `nn.ReLU(inplace=True)` for efficiency

## Dependencies

- Find actual minimum working version; document WHY
- Sync versions across requirements files
- Lazy import optional deps inside using function; helpful ImportError message

## Git

- PR titles should match release notes style; use milestones for backports
- Add license header to new files
- Update MD5s when test files change
- Separate logical changes into separate PRs
- Keep files < 500 LOC
