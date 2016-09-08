import pandas as pd
import numpy as np
import ee
import sys 
sys.path.append('/rheil/notebooks/pygee')
import gee_tools
import gee_random




###################################################
# Run full workflow
###################################################
def sim_flow(input_ic, coefs_csv, coefs_se_csv, n):
    # Add parameters to images
    draws_dict = {}
    for i in range(n):
        draw = draw_random_coefs(coefs_csv, coefs_se_csv)
        draw = draw.reset_index()
        draw = pd.melt(draw, id_vars = 'index')
        draw['coef_name'] = draw.apply(lambda row: 'c_' + row['index'] + '_' + str(row['variable']), axis = 1)
        draw = draw.set_index('coef_name')
        draw = draw.drop(['variable', 'index'], axis = 1)
        draws_dict[i] = draw.to_dict()['value']
    draws_dict = gee_tools.dict_to_eedict(draws_dict)
    c2p_mapper = coefs_to_params_map(draws_dict)
    input_ic = input_ic.map(c2p_mapper)
    
    # Generate random raster images
    template = ee.Image(input_ic.first()).select('olu')
    rand_imgs = gee_random.runiform_rasters(template, n)
    
    # Simulate conversion probabilities
    sim_probs_mapper = sim_probs_map(coefs_csv)
    pr_ic = input_ic.map(sim_probs_mapper)
    cum_pr_ic = pr_ic.map(gen_cum_img)
    cum_pr_ic = cum_pr_ic.combine(rand_imgs.select('Ran_uniform'))
    clas_ic = cum_pr_ic.map(assign_clas)
    mask_mapper = gee_tools.mask_update_map(ee.Image(input_ic.first()).select('mask').eq(0))
    clas_ic = clas_ic.map(mask_mapper)
    pr_ic = pr_ic.map(mask_mapper)
    cum_pr_ic = cum_pr_ic.map(mask_mapper)
    return pr_ic, cum_pr_ic, clas_ic


###################################################
# Load inputs and parameters
###################################################
class coefs_to_params_map:
    def __init__(self, param_dicts):
        self.param_dicts = param_dicts
    def __call__(self, img):
        i = img.get('system:index')
        param_dict = self.param_dicts.get(i)
        img = img.set(param_dict)
        return img    
    

def load_coefs(coefs_csv, time_period = 'pool'):
    """
    Parameters
    ----------
    coefs_csv: str
        path csv generated from Stata code
        
    time_period: str
        Indicates which set of coefficients to use, either 2001, 2011 or pool
    
    Returns
    -------
    coefs_df: pandas df
        Cleaned up pandas df to be used in simulation
    """
    coefs_df = pd.read_csv(coefs_csv, delimiter='\t', index_col = 0)
    coefs_df = coefs_df.replace(np.nan, 0)
    coefs_df = coefs_df.iloc[2:,:]
    coefs_df = coefs_df.drop([idx for idx in coefs_df.index if 'o.' in idx])
    coefs_dict={}
    coefs_dict['2001']=coefs_df[[col for col in coefs_df.columns if '2001' in col]]
    coefs_dict['2011']=coefs_df[[col for col in coefs_df.columns if '2011' in col]]
    coefs_dict['pool']=coefs_df[[col for col in coefs_df.columns if 'pool' in col]]
    coefs_df=coefs_dict['pool']
    coefs_df = coefs_df.rename(columns = lambda x: int(x[3:x.rfind('_')]))
    return coefs_df



def draw_random_coefs(coefs_csv, coefs_se_csv, time_period = 'pool'):
    """
    Parameters
    ----------
    coefs_csv: str
        path to csv of coefficient estimates generated from Stata code
       
    coefs_se_csv: str
        path to csv of se estimates generated from Stata code
        
    Returns
    -------
    rand_coefs_df: pandas df
        Dataframe containing 1 random draw from the coefficient distribution
    """
    coefs_df = load_coefs(coefs_csv, time_period)
    coefs_se_df = load_coefs(coefs_se_csv, time_period)
    rand_coefs_df = pd.DataFrame(np.random.normal(loc = coefs_df.astype(float), scale = coefs_se_df.astype(float)))
    rand_coefs_df = rand_coefs_df.astype(str)
    rand_coefs_df.index = coefs_df.index
    rand_coefs_df.columns = coefs_df.columns
    return rand_coefs_df



def sep_formulas(formula_list, lu_types):
    """
    Parameters
    ----------
    formula_list: list
        List containing text strings of formulas to execute for 
        simulation calculation
    
    lu_types: list
        List of land use stubs to be found in formula_list
    
    Returns
    -------
    formula_dict: dict
        Dict containing full prediction formula for each land use type
        
    """
    formula_dict={}
    for lu_type in lu_types:
        l=['(' + formula + ')' for formula in formula_list if lu_type in formula]
        s=' + '.join(l)
        s = s.replace("r_" + lu_type + "dum", '1.')
        formula_dict[lu_type] = s
    return formula_dict


def gen_formula_dicts(coefs_df, lu_types=['ag', 'plant', 'for']):
    """
    Checks to make sure data_dict aligns with model coefficients from stata.
    Generates formulas for prediction generation.
    
    Parameters
    ----------
    coefs_df: pandas dataframe
        Dataframe containing all coefficients for conditional logit model
        
    Returns
    -------
    formula_dict: dict
        Dict containing text strings of formulas to execute for 
        simulation calculation
    
    """
    # Prep formula
    coefs_list = [var.split('_') for var in coefs_df.index]
    formula_dict = {}
    for lu_from in [1, 3, 5, 19]:
        formula_list=[]
        for var_list in coefs_list:
            formula = ['r_' + var for var in var_list]
            formula = (' * '.join(formula))
            formula+=" * " + 'c_' + ('_'.join(var_list))
            formula_list.append(formula)
        formula_dict[lu_from] = sep_formulas(formula_list, lu_types)
    return formula_dict


def gen_input_dicts(coefs_df, input_img):
    """
    Generates dictionary used in ee.Image.expression that assigns image bands
    and properties to the variables called in equations generated in gen_formula_dicts.
    
    Parameters
    ----------
    coefs_df: pandas dataframe
        Dataframe containing all coefficients for conditional logit model
    
    input_img: ee.Image
        Image to apply expression to, must have proper properties and bands
        
    Returns
    -------
    input_dicts: dict
        Dict of dicts containing a dict of ee definitions of simulation inputs for 
        each starting land use
    """
    
    coefs_list = coefs_df.index
    band_list = [var.split('_') for var in coefs_list]
    band_list = [item for sublist in band_list for item in sublist]
    band_list = list(set(band_list))
    band_dict = {'r_' + band: input_img.select(band) for band in band_list} 
    coefs_dict = {}
    for lu_from in [1, 3, 5, 19]:
        coefs_dict[lu_from] = {'c_' + coef: ee.Number.parse(input_img.get('c_' + coef + '_' + str(lu_from))) for coef in coefs_list}
        coefs_dict[lu_from].update(band_dict)
    input_dicts = coefs_dict
    return input_dicts


###################################################
# Estimate conversion probabilities
###################################################
class sim_probs_map:
    def __init__(self, coefs_csv):
        self.coefs_df = load_coefs(coefs_csv)
    def __call__(self, input_img):
        pr_img = sim_probs(self.coefs_df, input_img)
        return pr_img    


def sim_probs(coefs_df, input_img):
    """
    Parameters
    ----------
    coefs_df: pd.DataFrame
        Stata multinomial logit regression coefficients, results from load_coefs or draw_random_coefs
    
    input_img: ee.Image
        Image with band names corresponding to variable used in coefs_df
        
    Returns
    -------
    pr_img: ee.Image
        Image with one probability band for each of the outcome land uses
    
    """
    formula_dicts = gen_formula_dicts(coefs_df)
    input_dicts = gen_input_dicts(coefs_df, input_img)
    img_dict = {}
    for lu_from, formula_dict in formula_dicts.items():
        input_dict = input_dicts[lu_from]
        img_dict[lu_from] = {}
        for lu_to, formula in formula_dict.items():
            img_dict[lu_from][lu_to] = input_img.expression(formula, input_dict).exp()
    exp_dict = {}
    for lu_to in formula_dict.keys():
        exp_dict[lu_to] = ee.Image.constant(ee.Number(0))
        exp_dict[lu_to] = exp_dict[lu_to].add(input_img.select('olu').eq(1).multiply(img_dict[1][lu_to]))\
            .add(input_img.select('olu').eq(3).multiply(img_dict[3][lu_to]))\
            .add(input_img.select('olu').eq(5).multiply(img_dict[5][lu_to]))\
            .add(input_img.select('olu').eq(19).multiply(img_dict[19][lu_to]))
    pr_dict = {}
    denom_img = ee.Image.constant(1.).add(exp_dict['ag']).add(exp_dict['for']).add(exp_dict['plant'])
    pr_dict['shrub'] = ee.Image.constant(1.).divide(denom_img)
    for lu_to in formula_dict.keys():
        pr_dict[lu_to] = exp_dict[lu_to].divide(denom_img)
    pr_img = pr_dict['for'].addBands(pr_dict['plant']).addBands(pr_dict['shrub']).addBands(pr_dict['ag'])
    pr_img = pr_img.select([0, 1, 2, 3], ['for', 'plant', 'shrub', 'ag'])
    return pr_img    


###################################################
# Assign conversions
###################################################
def gen_cum_img(pr_img):
    """
    Parameters
    ----------
    pr_img: ee.Image
        Multi-band probability image that should be converted into a cumulative sum image
    
    Returns
    -------
    cum_img: ee.Image
        Image of cumulative sum of bands from original img
    """
    cum_img = ee.Image()
    cum_img = cum_img.addBands(pr_img.select('for')).select('for')
    cum_img = cum_img.addBands(pr_img.select('plant').add(cum_img.select('for')))
    cum_img = cum_img.addBands(pr_img.select('shrub').add(cum_img.select('plant')))
    cum_img = cum_img.addBands(pr_img.select('ag').add(cum_img.select('shrub')))
    return cum_img


def assign_clas(cum_pr_img):
    """
    Parameters
    ----------
    cum_pr_img: ee.Image
        Image that combines bands from simulate_probs->gen_cum_img and gee_random.runiform_rasters
        
    Returns
    -------
    clas_img: ee.Image
        Image of assigned classes
    """
    clas_img = ee.Image()
    clas_img = cum_pr_img.select('Ran_uniform').lte(cum_pr_img.select('for')).multiply(1)\
        .add(cum_pr_img.select('Ran_uniform').gt(cum_pr_img.select('for'))\
             .And(cum_pr_img.select('Ran_uniform').lte(cum_pr_img.select('plant'))).multiply(3))\
        .add(cum_pr_img.select('Ran_uniform').gt(cum_pr_img.select('plant'))\
             .And(cum_pr_img.select('Ran_uniform').lte(cum_pr_img.select('shrub'))).multiply(5))\
        .add(cum_pr_img.select('Ran_uniform').gt(cum_pr_img.select('shrub'))\
             .And(cum_pr_img.select('Ran_uniform').lte(ee.Image.constant(1))).multiply(19))
    clas_img = clas_img.select(['Ran_uniform'], ['clas'])
    return clas_img





def dict_to_input_ic(asset_dict, n):

    input_img = ee.Image()
    input_img = input_img.addBands(list(asset_dict.values()))
    temp_bands = ['b1']
    temp_bands.extend(['b1_'+str(i+1) for i in range(len(asset_dict)-1)])
    input_img = input_img.select(temp_bands, list(asset_dict.keys()))
    mask = input_img.select('mask').eq(0)
    input_img.updateMask(mask)
    input_img_list = [input_img] * n
    input_ic = ee.ImageCollection(input_img_list)
    return input_ic
"""

class sim_mapper:

    def __init__(self, coefs_csv, coefs_se_csv, n):
        self.coefs_csv = coefs_csv
        self.coefs_se_csv = coefs_se_csv
        draws = []
        for i in range(n):
            draw = draw_random_coefs(coefs_csv, coefs_se_csv)
            draws.append(draw)
        self.draws = draws
    def __call__(self, input_img):
        i = input_img.get('system: index').getInfo()
        coefs_df = self.draws[i]
        printcoefs_df
        coefs_df = draw_random_coefs(self.coefs_csv, self.coefs_se_csv)
        pr_img = simulate_probs(coefs_df, input_img)
        return pr_img
"""