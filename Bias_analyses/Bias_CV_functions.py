import pandas as pd

def CV_methods(original_df,m1,m2,min_replicates=5):
    
    '''
    This function finds the spices with measured gm by both of the two given methods
    with a given number (or more) of replicates. It then calculates the 
    coefficient of variation (CV) of gm over the replicates of both methods. 

    Parameters
    ----------
    original_df: Pandas DataFrame
        the Excel file of the original data set provided by Knauer et al. 2022 published in the link: 
        https://doi.org/10.6084/m9.figshare.19681410.v1.
    m1 : str
        Name of the first given method from the column of methods in the original_df.
    m2 : str
        Name of the second given method from the column of methods in the original_df.
    min_replicates : int, optional
        The minimum number of replicates required for a species to calculate
        CV of its gm values. The default is 5.

    Returns
    -------
    res_df : Pandas DataFrame
        The CV results of the available species.

    '''
    df_m=original_df[['species', 'method', 'gm']].copy(deep=True)
    
    df_m['pair_count'] = df_m.groupby(['species', 'method']).transform('count')
    df_m = df_m[df_m['pair_count'] >= min_replicates]
        
    df_m1 = df_m[df_m['method']==m1].copy(deep=True)
    df_m2 = df_m[df_m['method']==m2].copy(deep=True)
    
    df_m1 = df_m1.groupby(['species', 'method'])['gm'].apply(lambda arr: arr.std()/arr.mean()).reset_index()
    df_m1 = df_m1.rename(columns={'gm': f'CV_{m1}'})

    df_m2 = df_m2.groupby(['species', 'method'])['gm'].apply(lambda arr: arr.std()/arr.mean()).reset_index()
    df_m2 = df_m2.rename(columns={'gm': f'CV_{m2}'})

    res_df = pd.merge(df_m1, df_m2,  on='species', how='inner')
    res_df=res_df.dropna()
    
    if res_df.empty: 
        raise ValueError(r'There is no species with avalilable data for the given methods!')
    else:
        return res_df

###############################################################################
def Correlation_methods(original_df,m1,m2,n_min=5):
    
    '''
    This function finds the spices with measured gm by both of the two given methods
    and then calculates the correlation between the gm values of these methods 
    over the available species.
    For those methods with more than one replicate on a single species, the average 
    of the gm values was computed. 

    Parameters
    ----------
    original_df: Pandas DataFrame
        the Excel file of the original data set provided by Knauer et al. 2022 published in the link: 
        https://doi.org/10.6084/m9.figshare.19681410.v1.
    m1 : str
        Name of the first given method from the column of 'method' in the original_df.
    m2 : str
        Name of the second given method from the column of 'method' in the original_df.
    n_min : int, optional
        The minimum number of data points (species) required for calculating correlations.
        
    Returns
    -------
    res_df : Pandas DataFrame
        The table of species together with their (average) gm values measured by 
        both of the given methods.  

    '''
    
    df_m=original_df[['species', 'method', 'gm']].copy(deep=True)
    df_m1 = df_m[df_m['method']==m1].copy(deep=True)
    df_m2 = df_m[df_m['method']==m2].copy(deep=True)
    
    df_m1 = df_m1.groupby(['species', 'method']).agg({'gm': 'mean'}).reset_index()
    df_m1 = df_m1.rename(columns={'gm': f'av_gm_{m1}'})
    
    df_m2 = df_m2.groupby(['species', 'method']).agg({'gm': 'mean'}).reset_index()
    df_m2 = df_m2.rename(columns={'gm': f'av_gm_{m2}'})
      
    res_df = pd.merge(df_m1, df_m2,  on='species', how='inner')
    res_df=res_df.dropna()
    
    if len(res_df)>=n_min: 
        correlation = res_df[f'av_gm_{m1}'].corr(res_df[f'av_gm_{m2}'])
        print(f"N = {len(res_df)}, Correlation between the methods {m1} and {m2} = {correlation}")
        return res_df
    else: 
        raise ValueError(f'N = {len(res_df)}, The number of species is less than the minimum required number for correlation!')
###############################################################################
def Correlation_methods_variants(original_df,m1,v1,m2,v2,n_min=5):
    '''
    This function finds the spices with measured gm by both of the two given paired method-variants
    and then calculates the correlation between the gm values of these method-variants 
    over the available species.
    For those method-variants with more than one replicate on a single species, the average 
    of the gm values was calculated. 

    Parameters
    ----------
    original_df: Pandas DataFrame
        the Excel file of the original data set provided by Knauer et al. 2022 published in the link: 
        https://doi.org/10.6084/m9.figshare.19681410.v1.
    m1 : str
        Name of the first given method from the column of 'method' in the original_df.
    v1 : str
        Name of the variant of the first given method from the column of 'variant'
        in the original_df.
    m2 : str
        Name of the second given method from the column of 'method' in the original_df.
    v2 : str
        Name of the variant of the second given method from the column of 'variant'
        in the original_df.
    n_min : int, optional
        The minimum number of data points (species) required for calculating correlations.
        
    Returns
    -------
    res_df : Pandas DataFrame
        The table of species together with their (average) gm values measured by 
        both of the given methods.  
        

    '''
    df_mv=original_df[['species', 'method','variant', 'gm']].copy(deep=True)
    df_m1v1 = df_mv[((df_mv['method'] == m1) & (df_mv['variant'] == v1))].copy(deep=True)
    df_m2v2 = df_mv[((df_mv['method'] == m2) & (df_mv['variant'] == v2))].copy(deep=True)
    
    df_m1v1 = df_m1v1.groupby(['species', 'method','variant']).agg({'gm': 'mean'}).reset_index()
    df_m1v1 = df_m1v1.rename(columns={'gm': f'av_gm_{m1}_{v1}'})
    df_m2v2 = df_m2v2.groupby(['species', 'method','variant']).agg({'gm': 'mean'}).reset_index()
    df_m2v2 = df_m2v2.rename(columns={'gm': f'av_gm_{m2}_{v2}'})
    
    res_df = pd.merge(df_m1v1, df_m2v2,  on=['species'], how='right')
    res_df=res_df.dropna()
    
    if len(res_df)>=n_min: 
        correlation = res_df[f'av_gm_{m1}_{v1}'].corr(res_df[f'av_gm_{m2}_{v2}'])
        print(f"N = {len(res_df)}, Correlation between the method {m1} (variant {v1}) and method {m2} (variant {v2}) = {correlation}")
        return res_df
    else: 
        raise ValueError(f'N = {len(res_df)}, The number of species is less than the minimum required number for correlation!')
