import pandas as pd
import numpy as np
import ee

def simulate_probs(coefs_csv, map_dict):
    """
    Parameters
    ----------
    coefs_df: str
        Path of stata multinomial logit regression coefficients
    
    map_dict: dict
        Assigns ee assets to variable names used in coefs_df
        
    Returns
    -------
    pr_img: ee.Image
        Image with one probability band for each of the outcome land uses
    
    """
    coefs_df = load_coefs(coefs_csv)
    formula_dict = gen_formulas_df(coefs_df)
    img_dict = {}
    for lu_from, formula_to_dict in formula_dict.items():
        formula_to_dict = formula_dict[lu_from]
        img_dict[lu_from] = {}
        for lu_to, formula in formula_to_dict.items():
            img_dict[lu_from][lu_to] = ee.Image().expression(formula, map_dict).exp()
    exp_dict = {}
    lu_tos = ['ag', 'for', 'plant']
    for lu_to in lu_tos:
        exp_dict[lu_to] = ee.Image.constant(0)
        exp_dict[lu_to] = exp_dict[lu_to].add(map_dict['olu'].eq(1).multiply(img_dict[1][lu_to]))\
            .add(map_dict['olu'].eq(3).multiply(img_dict[3][lu_to]))\
            .add(map_dict['olu'].eq(5).multiply(img_dict[5][lu_to]))\
            .add(map_dict['olu'].eq(19).multiply(img_dict[19][lu_to]))
    pr_dict = {}
    denom_img = ee.Image.constant(1.).add(exp_dict['ag']).add(exp_dict['for']).add(exp_dict['plant'])
    pr_dict['shrub'] = ee.Image.constant(1.).divide(denom_img)
    for lu_to in lu_tos:
        pr_dict[lu_to] = exp_dict[lu_to].divide(denom_img)
    pr_img = pr_dict['for'].addBands(pr_dict['plant']).addBands(pr_dict['shrub']).addBands(pr_dict['ag'])
    pr_img = pr_img.select([0, 1, 2, 3], ['for', 'plant', 'shrub', 'ag'])
    return pr_img


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
        s = s.replace(lu_type + 'dum[0]', '1.')
        formula_dict[lu_type] = s
    return formula_dict

def gen_formulas_df(coefs_df, lu_types=['ag', 'plant', 'for']):
    """
    Checks to make sure data_dict aligns with model coefficients from stata.
    Generates formulas for prediction generation.
    
    Parameters
    ----------
    coefs_df: pandas series
        Series containing all coefficients for conditional logit model
        
    Returns
    -------
    formula_list: list
        List containing text strings of formulas to execute for 
        simulation calculation
    
    data_dict: dict
        Updated data_dict including constants for dummy variables
    """
    coefs_list=[var.split('_') for var in coefs_df.index]
    formula_dict = {}
    for lu_from in [1, 3, 5, 19]:
        formula_list=[]
        for var_list in coefs_list:
            formula=[var + '[0]' for var in var_list]
            formula=' * '.join(formula)
            formula+=" * " + coefs_df.loc['_'.join(var_list), lu_from]
            formula_list.append(formula)
        formula_dict[lu_from] = sep_formulas(formula_list, lu_types)
    return formula_dict

