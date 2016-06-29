# -*- coding: utf-8 -*-
"""
GEE sugar to remove clouds and create quality mosaics

"""
import ee
ee.Initialize()

def maskCloudsLandsatSR(image):
    mask = image.select(['cfmask']).eq(0)
    return image.updateMask(mask)

def maskCloudsLandsatTOA(image):
    mask = image.select(['fmask']).eq(0)
    return image.updateMask(mask)

def maskCloudsSentinel(image):
    mask = ee.Image(0).where(~image.select('QA60').gte(1024), 1)
    return image.updateMask(mask)

def qualityLandsat(image):
    image = maskCloudsLandsatTOA(image)
    image = image.addBands(image.normalizedDifference(['B4', 'B3']))
    image = image.addBands(image.metadata('system:time_start'))
    return image

