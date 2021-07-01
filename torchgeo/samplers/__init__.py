from .samplers import GeoSampler, GridGeoSampler, RandomGeoSampler


# https://stackoverflow.com/questions/40018681
GeoSampler.__module__ = "torchgeo.samplers"
GridGeoSampler.__module__ = "torchgeo.samplers"
RandomGeoSampler.__module__ = "torchgeo.samplers"


__all__ = ("GeoSampler", "GridGeoSampler", "RandomGeoSampler")
