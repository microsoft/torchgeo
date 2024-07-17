#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

filename = '0123456-012345678901234.csv'

size = 6
data = {
    'gbifID': [''] * size,
    'datasetKey': [''] * size,
    'occurrenceID': [''] * size,
    'kingdom': ['Animalia'] * size,
    'phylum': ['Chordata'] * size,
    'class': ['Mammalia'] * size,
    'order': ['Primates'] * size,
    'family': ['Hominidae'] * size,
    'genus': ['Homo'] * size,
    'species': ['Homo sapiens'] * size,
    'infraspecificEpithet': [''] * size,
    'taxonRank': ['SPECIES'] * size,
    'scientificName': ['Homo sapiens Linnaeus, 1758'] * size,
    'verbatimScientificName': ['Homo sapiens Linnaeus, 1758'] * size,
    'verbatimScientificNameAuthorship': ['Linnaeus, 1758'] * size,
    'countryCode': ['US'] * size,
    'locality': ['Chicago'] * size,
    'stateProvince': ['Illinois'] * size,
    'occurrenceStatus': ['PRESENT'] * size,
    'individualCount': [1] * size,
    'publishingOrgKey': [''] * size,
    'decimalLatitude': [41.881832] * size,
    'decimalLongitude': [''] + [-87.623177] * (size - 1),
    'coordinateUncertaintyInMeters': [5] * size,
    'coordinatePrecision': [''] * size,
    'elevation': [''] * size,
    'elevationAccuracy': [''] * size,
    'depth': [''] * size,
    'depthAccuracy': [''] * size,
    'eventDate': ['', '', '', '', -450, '2022-04-16T10:13:35.123Z'],
    'day': [16, '', '', '', '', 16],
    'month': [4, '', '', 12, 4, 4],
    'year': [2022, '', 2022, 2022, 2022, 2022],
    'taxonKey': [1] * size,
    'speciesKey': [1] * size,
    'basisOfRecord': ['HUMAN_OBSERVATION'] * size,
    'institutionCode': [''] * size,
    'collectionCode': [''] * size,
    'catalogNumber': [''] * size,
    'recordNumber': [''] * size,
    'identifiedBy': [''] * size,
    'dateIdentified': [''] * size,
    'license': [''] * size,
    'rightsHolder': [''] * size,
    'recordedBy': [''] * size,
    'typeStatus': [''] * size,
    'establishmentMeans': [''] * size,
    'lastInterpreted': [''] * size,
    'mediaType': [''] * size,
    'issue': [''] * size,
}

df = pd.DataFrame(data)
df.to_csv(filename, sep='\t', index=False)
