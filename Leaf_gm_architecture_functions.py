import pandas as pd
import numpy as np
from itertools import combinations
####
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
#################################### 
####################################################################################
def data_aggregation(original_df):
    """ 
    This function reads the gm dataset from Knauer et al. 2002 and aggregates it such that:
    If gm was measured with more than one method in an identical study, it is calculated as the mean of all the methods.
    
     
    Note: In this function, we get the average of all the duplicated numeric values; however, it is initially necessary
    for us to have the average values only for gm. 
    The anatomical traits we used are supposed to be fixed in repetitions of the same experiment. Therefore, averaging 
    them will not change their values, even if they are written in only one of the repeated columns and the rest are empty. 
    To use this function for columns other than those we used in our models, the function must be localized based on the
    application and goal of aggregation.
    
    Parameters
    ----------
    original_df: Pandas DataFrame
        the Excel file of the original data set provided by Knauer et al. 2022 published in the link: 
        https://doi.org/10.6084/m9.figshare.19681410.v1.
    Returns 
    -------
    aggregated_df: Pandas DataFrame
        The aggregated data set.
    """
    df=original_df.copy(deep=True)
    original_columns_order = df.columns.tolist()
    numeric_columns = (df.select_dtypes(include=['number'])).columns.tolist()
    ###########################
    #############
    def custom_agg(series):
        if series.name in numeric_columns:
            return series.mean() 
        elif series.name in str_cols:
            series.str.cat(sep=', ')
            return series.str.cat(sep=', ')
        else:
            return series.iloc[0] 
    ###########################
    grouping_cols = ['refkey','species','cultivar_variety','population_year','growth_environment']
    str_cols = ['method','variant','method_reference']
    
    aggregated_df = df.groupby(grouping_cols,dropna=False).agg(custom_agg).reset_index()
    aggregated_df = aggregated_df[original_columns_order] 
    
    aggregated_df.to_excel('gm_dataset_Knauer_et_al_2022_aggregated.xlsx', index=False)
    ##########################
    return aggregated_df
###############################################################################
def make_all_possible_combinations(t_list):
    """ 
    Constructing all the possible combinations of the given traits.
    
    Parameters
    ----------
    t_list : list
        List of the traits to be analyzed, including one or a group of existing traits (column names) in the
        Knauer et al. 2002 dataset.
        
    Returns 
    -------
    ls_combinations : list
        List of tuples, each containing a possible combination of traits.
    """
    ls_combinations = []
    for n in range(len(t_list)):
        ls_combinations += list(combinations(t_list, n+1))
    return(ls_combinations)
###############################################################################
def RF_with_spilit(comb_df,ensemble,minimum_data):
    """ 
    Training an ensemble of random forest models by randomly splitting the data to 70% train and 30% test sets.
    
    Parameters
    ----------
    comb_df : Pandas DataFrame
        The dataframe with specified PFT contains the column 'gm' and a column for each trait in the combination.
        Note: The first column necesserily must be for 'gm'.
    ensemble : integer
        The number of repeats of the model with a different random_state.
    minimum_data : integer
        The minimum number of data samples (raws) that data must contain before splitting.
        
    Returns 
    -------
    res : dict 
        The resulting scores containing::
            R2: The average coefficient of determination over different models.
            R2_err: The standard error for average over the coefficient of determination of different models.
            R2_adj: The average adjusted coefficient of determination over different models.
            R2_adj_err: The standard error for the average adjusted coefficient of determination over different models.
            r: The average Pearson correlation of determination over different models.
            r_err: The standard error for the average Pearson correlation of determination over different models.
            importances : The list of the importance of the traits in the model.
    
    """
    if comb_df.shape[0]<minimum_data: 
        print('The number of data points is less than miniumum!')
        res={'R2': np.nan,'R2_err': np.nan,'R2_adj': np.nan,'R2_adj_err': np.nan,
             'r': np.nan,'r_err': np.nan,'importances': np.nan}
    else:
        X= np.array(comb_df.values)[:,1:] 
        y= np.array(comb_df.values)[:,0] 
        ###
        r2s=[]
        r2_adjs=[]
        corrs=[]
        imps=[]
        for _ in range(ensemble):
            model = RandomForestRegressor(n_estimators=100, random_state=None)
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=None)
            model.fit(X_train, y_train)
            y_pred = model .predict(X_test)
            r2=r2_score(y_test, y_pred)
            r2s.append(r2)
            ####
            r2_adj= 1 - (((1 - r2) * (X_test.shape[0] - 1)) / (X_test.shape[0] - X_test.shape[1] - 1))
            r2_adjs.append(r2_adj)
            ####
            corrs.append(stats.pearsonr(y_test, y_pred)[0])
            ####
            imps.append(np.array(model.feature_importances_))
        imps_mean=np.mean(np.array(imps),axis=0)
        res={'R2': np.mean(r2s),'R2_err': stats.sem(r2s),'R2_adj': np.mean(r2_adjs),'R2_adj_err': stats.sem(r2_adjs) ,
             'r': np.mean(corrs),'r_err': stats.sem(corrs),'importances': imps_mean/np.sum(imps_mean)}
    return res  
###############################################################################
def RF_with_train_and_test_data(comb_df_train,comb_df_test,ensemble,minimum_train_data,minimum_test_data):
    """ 
    Training an ensemble of random forest models with differnet random states for the given fixed traing and test data.
    
    Parameters
    ----------
    comb_df_train : Pandas DataFrame
        The dataframe of train set with specified PFT contains the column 'gm' and a column for each trait in the combination.
        Note: The first column necesserily must be for 'gm'. 
    comb_df_tset : Pandas DataFrame
        The dataframe of test set with specified PFT contains the column 'gm' and a column for each trait in the combination.
        Note: The first column necesserily must be for 'gm'. 
    ensemble : integer
        The number of repeats of the model with a different random_state.
    minimum_train_data : integer
        The minimum number of data samples (rows) that the train set must contain.
    minimum_test_data : integer
        The minimum number of data samples (rows) that the train set must contain.
        
    Returns 
    -------
    res : dict 
        The resulting scores containing::
            R2: The average coefficient of determination over different models.
            R2_err: The standard error for average over the coefficient of determination of different models.
            R2_adj: The average adjusted coefficient of determination over different models.
            R2_adj_err: The standard error for the average adjusted coefficient of determination over different models.
            r: The average Pearson correlation of determination over different models.
            r_err: The standard error for the average Pearson correlation of determination over different models.
            importances : The list of the importance of the traits in the model.
    
    """
    if comb_df_train.shape[0]<minimum_train_data or comb_df_test.shape[0]<minimum_test_data or comb_df_train.shape[0]<comb_df_test.shape[0]: 
        print('The number of data points is less than miniumum!')
        res={'R2': np.nan,'R2_err': np.nan,'R2_adj': np.nan,'R2_adj_err': np.nan,
             'r': np.nan,'r_err': np.nan,'importances': np.nan}
    else:
        X_train= np.array(comb_df_train.values)[:,1:] 
        y_train= np.array(comb_df_train.values)[:,0] 
        X_test= np.array(comb_df_test.values)[:,1:] 
        y_test= np.array(comb_df_test.values)[:,0] 
        ###
        r2s=[]
        r2_adjs=[]
        corrs=[]
        imps=[]
        for _ in range(ensemble):
            model = RandomForestRegressor(n_estimators=100, random_state=None)
            model.fit(X_train, y_train)
            y_pred = model .predict(X_test)
            r2=r2_score(y_test, y_pred)
            r2s.append(r2)
            ####
            r2_adj= 1 - (((1 - r2) * (X_test.shape[0] - 1)) / (X_test.shape[0] - X_test.shape[1] - 1))
            r2_adjs.append(r2_adj)
            ####
            corrs.append(stats.pearsonr(y_test, y_pred)[0])
            ####
            imps.append(np.array(model.feature_importances_))
        imps_mean=np.mean(np.array(imps),axis=0)
        res={'R2': np.mean(r2s),'R2_err': stats.sem(r2s),'R2_adj': np.mean(r2_adjs),'R2_adj_err': stats.sem(r2_adjs) ,
             'r': np.mean(corrs),'r_err': stats.sem(corrs),'importances': imps_mean/np.sum(imps_mean)}
    return res  
###############################################################################
def make_PFT_df(df,PFT):
    """ 
    Make the date set for the given PFT by removing the data points for all the other PFTs.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT : list
        The PFT of interest, including one or a group of existing names in the column 'plant_functional_type' of df.
        
    Returns 
    -------
    new_df : Pandas DataFrame 
        The new data set containing the data points only for the given PFT.
    
    """
    new_df = df[df['plant_functional_type'].isin(PFT)].copy(deep=True)
    return new_df
###############################################################################
def make_non_overlapping_PFT_df(df,PFT):
    """ 
    Make the date set which contains data points of all the PFTs except the given PFT.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT : list
        The PFT of interest, including one or a group of existing names in the column 'plant_functional_type' of df.
        
    Returns 
    -------
    new_df : Pandas DataFrame 
        The new data set containing all the data points except for the given PFT.
    
    """
    new_df = df[~df['plant_functional_type'].isin(PFT)].copy(deep=True)
    return new_df
###############################################################################
def make_combination_df(df,t_combination):
    """ 
    Make the date set for the given combination of traits by removing the data points for all the other traits.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    t_combination : list
        The combination of interest, including one or a group of existing traits (column names) in the df.
        
    Returns 
    -------
    new_df : Pandas DataFrame 
        The new data set containing the data points only for the given PFT.
    
    """
    new_df = df.loc[:,['gm']+t_combination].copy(deep=True)
    new_df = new_df.dropna(how='any')
    return new_df
###############################################################################
def trait_pairs_correlation (df_agg,traita1, trait2):
    """ 
    Get the correlation between two traits based on all the available data for this pair in the data set.    
    Parameters
    ----------
    df_agg : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    trait1 : list
        The name of the first trait of interest from the existing traits (column names) in the df_agg.
    trait2 : list
        The name of the second trait of interest from the existing traits (column names) in the df_agg.
        
    Returns 
    -------
    pearson_r :  float 
        The Pearson correlation between the traits in the available data.
    p_value :  float 
            The p-value of the Pearson correlation between the traits in the available data.
    
    """
    df_of_traits=df_agg.loc[:,[traita1, trait2]].copy(deep=True)
    df_of_traits=df_of_traits.dropna(how='any')
    traits_arr=df_of_traits.to_numpy()
    correlation=stats.pearsonr(traits_arr[:,0],traits_arr[:,1])
    pearson_r=correlation[0]
    p_value = correlation[1]
    return pearson_r,p_value
    
    
###############################################################################
def CV_with_PFT_and_combination_of_interest(df_agg,PFT_of_interest,combination_of_interest,
                                               enseble_size,min_rows=50):
    """ 
    To get the repeated cross-validation predictability scores and Gini importance of the traits for the PFT and combination of interest.
    
    Parameters
    ----------
    df_agg : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_of_interest : list
        The PFT of interest, including one or a group of existing names in the column 'plant_functional_type' of df.
    combination_of_interest: list
        The combination of interest, including one or a group of existing traits (column names) in the df.
    enseble_size: integer
        The number of executions for i) splitting data to train and test sets and ii) training the model with a different
        random state.
    min_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest must contain. 
        
    Returns 
    -------
    res : dict 
        The resulting predictability scores and importance of the given traits.
    """
    
    PFT_df=make_PFT_df(df_agg,PFT_of_interest)
    combination_df=make_combination_df(PFT_df,combination_of_interest)
    train_res=RF_with_spilit(combination_df,enseble_size,min_rows)
    return train_res
###############################################################################
def CV_with_PFT_of_interest(df_agg,PFT_of_interest,traits_list,enseble_size,min_rows=50):
    
    """ 
    To get the repeated cross-validation predictability scores and Gini importance of the traits for the PFT of interest
    and all the possible combinations of traits.
    
    Parameters
    ----------
    df_agg : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_of_interest : list
        The PFT of interest, including one or a group of existing names in the column 'plant_functional_type' of df.
    traits_list: list
        One or a group of existing traits (column names) in the df to be analyzed.
    enseble_size: integer
        The number of executions for i) splitting data to train and test sets and ii) training the model with a different
        random state.
    min_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest must contain. 
        
    Returns 
    -------
    res : Pandas DataFrame
        The DataFrame (table) of the resulting predictability scores and Gini importance of the traits for all available combinations.
    """
    
    PFT_df=make_PFT_df(df_agg,PFT_of_interest)
    trait_combinations=make_all_possible_combinations(traits_list)
    headers=['Traits','N','R2','R2_err','R2_adj','R2_adj_err','r','r_err','importances']
    table_df = pd.DataFrame(columns=headers)
    n_models=0
    for i, traits in enumerate(trait_combinations):
        combination_df=make_combination_df(PFT_df,list(traits))
        if combination_df.shape[0] < min_rows: continue 
        n_models += 1
        train_res=RF_with_spilit(combination_df,enseble_size,min_rows)   
        table_df.loc[n_models-1] = [list(traits)] + [combination_df.shape[0]] + [train_res[key] for key in headers[2:]]
    ########
    return table_df
###############################################################################
###############################################################################
def cross_prediction_global_PFT_with_combination_of_interest(df_agg,PFT_of_interest,combination_of_interest,
                                            enseble_size=5,minimum_train_rows=40,minimum_test_rows=10):
    
    """ 
    To get the predictability scores and Gini importance of the traits for cross-prediction between the (non-overlapping)
    global set and PFT of interest, for the given combination of traits.
    
    Parameters
    ----------
    df_agg : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_of_interest : list
        The PFT of interest, including one or a group of existing names in the column 'plant_functional_type' of df.
    combination_of_interest: list
        The combination of interest, including one or a group of existing traits (column names) in the df.
    enseble_size: integer
        The number of executions for training the model with a different random state.
    minimum_train_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the train set. 
    minimum_test_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the test set. 
        
    Returns 
    -------
    res : dict 
        The resulting predictability scores and Gini importance of the given traits.
    """
    
    PFT_df_train = make_non_overlapping_PFT_df(df_agg,PFT_of_interest)
    PFT_df_test = make_PFT_df(df_agg,PFT_of_interest)
    
    combination_df_train=make_combination_df(PFT_df_train,combination_of_interest)
    combination_df_test=make_combination_df(PFT_df_test,combination_of_interest)
    train_res=RF_with_train_and_test_data(combination_df_train,combination_df_test,enseble_size,
                                          minimum_train_rows,minimum_test_rows)   
    return train_res
###############################################################################
def cross_prediction_global_PFT(df_agg,PFT_of_interest,traits_list,
                                            enseble_size=150,minimum_train_rows=40,minimum_test_rows=10):
    
    """ 
    To get the predictability scores and Gini importance of the traits for cross-prediction between the (non-overlapping)
    global set and PFT of interest, for all the possible combinations of traits.
    
    Parameters
    ----------
    df_agg : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_of_interest : list
        The PFT of interest, including one or a group of existing names in the column 'plant_functional_type' of df.
    enseble_size: integer
        The number of executions for training the model with a different random state.
    minimum_train_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the train set. 
    minimum_test_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the test set. 
        
    Returns 
    -------
    res : Pandas DataFrame
        The DataFrame (table) of the resulting predictability scores and Gini importance of the traits for all available combinations.
    """
    
    PFT_df_train = make_non_overlapping_PFT_df(df_agg,PFT_of_interest)
    PFT_df_test = make_PFT_df(df_agg,PFT_of_interest)
    
    trait_combinations=make_all_possible_combinations(traits_list)
    headers=['Traits','N','R2','R2_err','R2_adj','R2_adj_err','r','r_err','importances']
    table_df = pd.DataFrame(columns=headers)
    n_models=0
    for i, traits in enumerate(trait_combinations):
        combination_df_train=make_combination_df(PFT_df_train,list(traits))
        combination_df_test=make_combination_df(PFT_df_test,list(traits))
        if combination_df_train.shape[0]<minimum_train_rows or combination_df_test.shape[0]<minimum_test_rows:  continue
        if combination_df_train.shape[0]<combination_df_test.shape[0]:  continue
        n_models += 1
        train_res=RF_with_train_and_test_data(combination_df_train,combination_df_test,enseble_size,
                                              minimum_train_rows,minimum_test_rows)   
        table_df.loc[n_models-1] = [list(traits)] + [[combination_df_train.shape[0],combination_df_test.shape[0]]]+ [train_res[key] for key in headers[2:]]
    ########
    return table_df
###############################################################################
###############################################################################
def available_PFT_pairs(df_agg,PFT_dict,traits_list,minimum_train_rows=40,minimum_test_rows=10):
    """
    To find all possible pairs of PFTs that do not overlap and include at least one combination with 
    the minimum required data for train and test sets. 
    Note: given we have 18 PFTs, 324 cross-prediction scenarios can be defined between different pairs of them. 
    To exclude the pairs which have overlap with each other or the ones that do not have minimum test or train data, 
    this function can be used.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_dict : dict
        The dictionary that includes the names of all the PFTs as the keys and the corresponding list 
        for each PFT as the values.
    traits_list: list
        List of the traits to be analyzed, including one or a group of existing traits (column names) in the df_agg.
    minimum_train_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the train set. 
    minimum_test_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the test set. 
        
    Returns 
    -------
    res : list
        List of pairs of PFTs that do not overlap and include at least one combination with the minimum required
        data for train and test sets.
    """
    trait_combinations = make_all_possible_combinations(traits_list)
    PFT_names= list(PFT_dict.keys())
    NO_pairs = []
    available_pairs = {}
    for pf1 in PFT_names:
        for pf2 in PFT_names:
            list1=PFT_dict[pf1]
            list2=PFT_dict[pf2]
            intersection = set(list1) & set(list2)
            if len(intersection)==0:
                NO_pairs.append([pf1,pf2])
                PFT_df_1= make_PFT_df(df_agg,list1)
                PFT_df_2= make_PFT_df(df_agg,list2)
                n_models=0
                for i, traits in enumerate(trait_combinations):
                    combination_df_train=make_combination_df(PFT_df_1,list(traits))
                    combination_df_test=make_combination_df(PFT_df_2,list(traits))
                    if combination_df_train.shape[0]<minimum_train_rows or combination_df_test.shape[0]<minimum_test_rows:  continue
                    if combination_df_train.shape[0]<combination_df_test.shape[0]:  continue
                    n_models += 1
                if n_models>0: available_pairs[pf1,pf2]=n_models
    return available_pairs
    
###############################################################################
###############################################################################
def cross_prediction_PFT_PFT_with_combination_of_interest(df_agg,PFT_train,PFT_test,combination_of_interest,
                                            enseble_size=50,minimum_train_rows=40,minimum_test_rows=10):
    
    """ 
    To get the predictability scores and Gini importance of the traits for cross-prediction between two PFTs of interest,
    for the given combination of traits.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_train : list
        The PFT of interest for training the model, including one or a group of existing names in the column 'plant_functional_type' of df.
    PFT_test : list
        The PFT of interest for tasting the model, including one or a group of existing names in the column 'plant_functional_type' of df.
    combination_of_interest: list
        The combination of interest, including one or a group of existing traits (column names) in the df.
    enseble_size: integer
        The number of executions for training the model with a different random state.
    minimum_train_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the train set. 
    minimum_test_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the test set. 
        
    Returns 
    -------
    res : dict 
        The resulting predictability scores and Gini importance of the given traits.
    """
    
    PFT_df_train = make_PFT_df(df_agg,PFT_train)
    PFT_df_test  = make_PFT_df(df_agg,PFT_test)
    
    combination_df_train=make_combination_df(PFT_df_train,combination_of_interest)
    combination_df_test=make_combination_df(PFT_df_test,combination_of_interest)
    train_res=RF_with_train_and_test_data(combination_df_train,combination_df_test,enseble_size,
                                          minimum_train_rows,minimum_test_rows)    
    return train_res
###############################################################################
def cross_prediction_PFT_PFT(df_agg,PFT_train,PFT_test,traits_list,
                                            enseble_size=150,minimum_train_rows=40,minimum_test_rows=10):
    
    """ 
    To get the predictability scores and Gini importance of the traits for cross-prediction between two PFTs of
    interest in all the possible combinations of traits.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The aggregated DataFrame containing the data set provided by Knauer et al. 2022.
    PFT_train : list
        The PFT of interest for training the models, including one or a group of existing names in the column 'plant_functional_type' of df.
    PFT_test : list
        The PFT of interest for tasting the models, including one or a group of existing names in the column 'plant_functional_type' of df.
    enseble_size: integer
        The number of executions for training the model with a different random state.
    minimum_train_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the train set. 
    minimum_test_rows: integer
        The minimum number of data sets (rows) that the available data for the PFT and combination of interest
        must contain in the test set. 
        
    Returns 
    -------
    res : Pandas DataFrame
        The DataFrame (table) of the resulting predictability scores and Gini importance of the traits for all available combinations.
    """
    
    PFT_df_train = make_PFT_df(df_agg,PFT_train)
    PFT_df_test  = make_PFT_df(df_agg,PFT_test)
    
    trait_combinations=make_all_possible_combinations(traits_list)
    headers=['Traits','N','R2','R2_err','R2_adj','R2_adj_err','r','r_err','importances']
    table_df = pd.DataFrame(columns=headers)
    n_models=0
    for i, traits in enumerate(trait_combinations):
        combination_df_train=make_combination_df(PFT_df_train,list(traits))
        combination_df_test=make_combination_df(PFT_df_test,list(traits))
        if combination_df_train.shape[0]<minimum_train_rows or combination_df_test.shape[0]<minimum_test_rows:  continue
        if combination_df_train.shape[0]<combination_df_test.shape[0]:  continue
        n_models += 1
        train_res=RF_with_train_and_test_data(combination_df_train,combination_df_test,enseble_size,
                                              minimum_train_rows,minimum_test_rows)   
        table_df.loc[n_models-1] = [list(traits)] + [[combination_df_train.shape[0],combination_df_test.shape[0]]]+ [train_res[key] for key in headers[2:]]
    ########
    return table_df
###############################################################################
###############################################################################
def total_importances(df_of_results):
    """
    To compute the total importance measures of all the traits contributing to the trained models of the given PFT.
    Parameters
    ----------
    df_of_results : Pandas DataFrame
        The DataFrame (table) of the resulting predictability scores and Gini importance of the traits for all
        available combinations for the prediction scenario of interest.
        
    Note: This function will return the total importance values only if the df_of_results of results contain 
        at least one model with an R2_adj>0. Otherwise, it will return empty DataFrames.
        
    Returns 
    -------
    IMP_G : Pandas DataFrame
        The total Gini importance of the contributing traits in the models.
    IMP_C : Pandas DataFrame
        The total contribution importance of the contributing traits in the models.
    """ 
    Res_DF=df_of_results.copy(deep=True) 
    Res_DF = Res_DF.sort_values(by='R2_adj', ascending=False)
    Res_DF = Res_DF[Res_DF['R2_adj'] > 0.0]
    ##############################
    traits0 = Res_DF['Traits'].tolist()
    flattened_list = [item for sublist in traits0 for item in sublist]
    unique_traits = list(set(flattened_list))
    for t in unique_traits:  
        Res_DF [t] = [float('nan')] * Res_DF.shape[0] 
    ##########
    imps_table_1 = Res_DF[unique_traits+['R2_adj']].copy(deep=True)
    imps_table_2 = Res_DF[unique_traits+['R2_adj']].copy(deep=True)
    ##############################
    for i in range(Res_DF.shape[0]):
        row_index=imps_table_1.index[i]
        trits_of_row=Res_DF.loc[row_index,'Traits']
        imps_of_row=Res_DF.loc[row_index,'importances']
        r2_adj_of_row=Res_DF.loc[row_index,'R2_adj']
        for j in range(len(trits_of_row)):
            imps_table_1.loc[row_index,trits_of_row[j]] = imps_of_row[j] * r2_adj_of_row 
            imps_table_2.loc[row_index,trits_of_row[j]] = r2_adj_of_row
    ##############################
    IMP_G = imps_table_1[unique_traits].mean(skipna=True).to_frame().T
    IMP_G = IMP_G.div(IMP_G.sum(axis=1), axis=0)
    #########
    IMP_C = imps_table_2[unique_traits].mean(skipna=True).to_frame().T
    IMP_C = IMP_C.div(IMP_C.sum(axis=1), axis=0)
    ##############################
    return IMP_G, IMP_C    
###############################################################################
