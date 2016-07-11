import pandas as pd
import numpy as np
import ee

class runiform_raster:
    def generateRan(self, seed):
        state1 = ee.Image(seed).select('Ran_state')
        state2 = state1.bitwiseXor(state1.leftShift(13))
        state3 = state2.bitwiseXor(state2.rightShift(17))
        state4 = state3.bitwiseXor(state3.leftShift(5))
        state5 = state4.multiply(1597334677).mod(4294967296)
        state6 = state4.addBands(state5.divide(4294967296).select([0],['Ran_uniform']))
        return state6.set('Ran_has_seed', True);
    
    def makeSeed(self, img):
        template = ee.Image(img)
        seed0 = ee.Image.pixelCoordinates(template.projection()).int32().add(2147483648).uint32()
        seed1 = seed0.select(0).bitwiseAnd(65535).leftShift(16).add(seed0.select(1).bitwiseAnd(65535))
        seed2 = seed1.add(seed1.eq(0).multiply(0x7FFF7FFF)).uint32()
        seed3 = seed2.select([0],['Ran_state']).updateMask(template.mask())
        for i in range(10):
          seed3 = self.generateRan(seed3)
        return seed3
  
    def __call__(self, img):
        img = ee.Image(img)
        return ee.Image(ee.Algorithms.If(img.get('Ran_has_seed'), self.generateRan(img), self.makeSeed(img)))

    
def runiform_rasters(template, n):
    ic = ee.ImageCollection(ee.List(ee.List.sequence(1,n).iterate(add_rand_img, [template])).slice(1))
    return ic


def add_rand_img(indx, col):
    runi = runiform_raster()
    col = ee.List(col)
    return col.add(runi(col.get(-1)))
    
## NOTE: If i want to add uniform, can adapt code from https://groups.google.com/forum/#!searchin/google-earth-engine-developers/random/google-earth-engine-developers/Oi0OAKnr4Vk/_a_4buqFAgAJ