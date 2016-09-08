# -*- coding: utf-8 -*-
"""
GEE sugar to remove clouds and create quality mosaics
"""

import ee

class prep_ic:
    def __init__(self, sensor, product):
        """
        Define the sensor and data product to call the appropriate parameters.
        """
        if sensor not in list(band_dict.keys()):
            raise ValueError('Invalid sensor id')
        if product not in list(cloud_dict.keys()):
            raise ValueError('Invalid data product id')
           
        self.b_dict = band_dict[sensor]
        self.cloud_key = cloud_dict[product]
        
        
    def __call__(self, img):
        """
        Used to map the initialized class onto an imagecollection. Will mask out clouds,
        add an ndvi band, and rename all bands.
        """
        # Mask out flagged clouds
        mask = eval('img.select([self.cloud_key[0]])' + \
                    self.cloud_key[1])
        img = img.updateMask(mask)
        
        # Rename bands
        old_names = list(self.b_dict.keys())
        new_names = list(self.b_dict.values())
        img = img.select(old_names, new_names)

        # Add ndvi
        img = img.addBands(img.normalizedDifference(['nir', 'red']))
        
        # Rename ndvi
        newer_names = new_names.copy()
        newest_names = new_names.copy()
        newer_names.append('nd')
        newest_names.append('ndvi')
        img = img.select(newer_names, newest_names)
    
        return img

### Define band names for different sensors
band_dict = {}
band_dict['l8'] =  {'B1': 'aerosol',
                    'B2': 'blue',
                    'B3': 'green',
                    'B4': 'red',
                    'B5': 'nir',
                    'B6': 'swir1',
                    'B7': 'swir2',
                    'B8': 'pan',
                    'B9': 'cirrus',
                    'B10': 'tir1',
                    'B11': 'tir2'} 

band_dict['l5'] =  {'B1': 'blue',
                    'B2': 'green',
                    'B3': 'red',
                    'B4': 'nir',
                    'B5': 'swir1',
                    'B6': 'tir1',
                    'B7': 'swir2'}

band_dict['s2'] =  {'B1': 'aerosol',
                    'B2': 'blue',
                    'B3': 'green',
                    'B4': 'red',
                    'B5': 're1',
                    'B6': 're2',
                    'B7': 're3',
                    'B8': 'nir',
                    'B8A': 're4',
                    'B9': 'wv',
                    'B10': 'cirrus',
                    'B11': 'swir1',
                    'B12': 'swir2'}

### Define masking procedure for different data products
cloud_dict = {}
cloud_dict['sr'] = ['cfmask', '.eq(0)']
cloud_dict['toa_fmask'] = ['fmask', '.eq(0)']
cloud_dict['sentinel'] = ['QA60', '.gte(1024)']



