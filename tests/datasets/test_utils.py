import pathlib

from torchgeo.datasets.utils import working_dir


def test_existing_directory(tmp_path: pathlib.Path) -> None:
    subdir = tmp_path / "foo" / "bar"
    subdir.mkdir(parents=True)

    assert subdir.exists()

    with working_dir(str(subdir)):
        assert subdir.cwd() == subdir


def test_nonexisting_directory(tmp_path: pathlib.Path) -> None:
    subdir = tmp_path / "foo" / "bar"

    assert not subdir.exists()

    with working_dir(str(subdir), create=True):
        assert subdir.cwd() == subdir
