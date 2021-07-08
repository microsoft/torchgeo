from .samplers import GeoSampler, GridGeoSampler, RandomGeoSampler

__all__ = ("GeoSampler", "GridGeoSampler", "RandomGeoSampler")

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.samplers"
