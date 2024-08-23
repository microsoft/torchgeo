# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PRISMA datasets."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .geo import RasterDataset
from .utils import Sample, percentile_normalization


class PRISMA(RasterDataset):
    """PRISMA dataset.

    Hyperspectral Precursor and Application Mission
    `PRISMA <https://www.eoportal.org/satellite-missions/prisma-hyperspectral>`__
    (PRecursore IperSpettrale della Missione Applicativa) is a medium-resolution
    hyperspectral imaging satellite, developed, owned, and operated by the
    Italian Space Agency `ASI <https://www.asi.it/en/earth-science/prisma/>`_
    (Agenzia Spaziale Italiana). It is the successor to the discontinued HypSEO
    (Hyperspectral Satellite for Earth Observation) mission.

    PRISMA carries two sensor instruments, the HYC (Hyperspectral Camera) module and the
    PAN (Panchromatic Camera) module. The HYC sensor is a prism spectrometer for two
    bands, VIS/NIR (Visible/Near Infrared) and NIR/SWIR (Near Infrared/Shortwave
    Infrared), with a total of 237 channels across both bands. Its primary mission
    objective is the high resolution hyperspectral imaging of land, vegetation,
    inner waters and coastal zones. The second sensor module, PAN, is a high resolution
    optical imager, and is co-registered with HYC data to allow testing of image fusion
    techniques.

    The HYC module has a spatial resolution of 30 m and operates in two bands, a 66
    channel VIS/NIR band with a spectral interval of 400-1010 nm, and a 171 channel
    NIR/SWIR band with a spectral interval of 920-2505 nm. It uses a pushbroom scanning
    technique with a swath width of 30 km, and a field of regard of 1000 km either side.
    The PAN module also uses a pushbroom scanning technique, with identical swath width
    and field of regard but spatial resolution of 5 m.

    PRISMA is in a sun-synchronous orbit, with an altitude of 614 km, an inclination
    of 98.19° and its LTDN (Local Time on Descending Node) is at 1030 hours.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2018.8517785

    .. note::
       PRISMA imagery is distributed as HDF5 files. However, TorchGeo does not yet have
       support for reprojecting and windowed reading of HDF5 files. This data loader
       requires you to first convert all files from HDF5 to GeoTIFF using something like
       `this script
       <https://gist.github.com/adamjstewart/7f5324a6a339c20e778f39536402fb4a>`__.

    .. versionadded:: 0.6
    """

    # https://prisma.asi.it/missionselect/docs/PRISMA%20Product%20Specifications_Is2_3.pdf
    #
    # See sections:
    #
    # * 6.3.6: L0A Product Naming Convention
    # * 7.5:   L1 Product Naming Convention
    # * 7.8.5: FKDP, GKDP, ICU-KDP and CDP Products Naming Convention
    filename_glob = 'PRS_*'
    filename_regex = r"""
        ^PRS
        _(?P<level>[A-Z\d]+)
        _(?P<product>[A-Z]+)
        (_(?P<order>[A-Z_]+))?
        _(?P<start>\d{14})
        _(?P<stop>\d{14})
        _(?P<version>\d{4})
        (_(?P<valid>\d))?
        \.
    """
    date_format = '%Y%m%d%H%M%S'

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # RGB band indices based on https://doi.org/10.3390/rs14164080
        rgb_indices = [34, 23, 11]
        image = sample['image'][rgb_indices].permute(1, 2, 0).float()
        image = percentile_normalization(image, axis=(0, 1))

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title('Image')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
