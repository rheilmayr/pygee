import pandas as pd
import numpy as np
import ee


def convert_to_feature(geo):
    """
    """
    return ee.Feature(ee.Geometry(geo))


def dissolve_groups(groupObject):
    """
    """
    group = ee.Dictionary(groupObject)
    geometries = ee.List(group.get('list'))
    features = ee.FeatureCollection(geometries.map(convert_to_feature))
    union = features.union().first()
    return union.set('id_1', group.get('group'))


def dict_to_eedict(d):
    """
    Recursive function to convert a python dict of arbitrary depth to an ee.Dictionary object
    """
    if not isinstance(d, dict) or not d:
        return d
    else:
        return ee.Dictionary({str(k): dict_to_eedict(v) for k, v in d.items()})

    
class band_renamer:
    """
    Using map, can rename bands of all images within an imagecollection
    """
    def __init__(self, orig_bandnames, new_bandnames):
        self.orig_bandnames = orig_bandnames
        self.new_bandnames = new_bandnames
    def __call__(self, img):
        img = img.select(self.orig_bandnames, self.new_bandnames)
        return img

    
def band_aggregator(new_band, full_img):
    """
    Using iterate, converts an image collection into a single multiband image.
    Purpose is to prepare the image for reduceRegions
    """
    full_img = ee.Image(full_img).addBands(new_band)
    return full_img


class mask_update_map:
    """
    Using map, can update mask for all images within an imagecollection
    """
    def __init__(self, mask):
        self.mask = mask
    def __call__(self, img):
        img = img.updateMask(self.mask)
        return img

    
