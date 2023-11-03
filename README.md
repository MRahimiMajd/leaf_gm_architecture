# leaf_gm_architecture
This file contains the instructions for using the shared Python functions to reproduce the results presented in the Rahimi-Majd et al. paper.

To run the functions, first, the Excell file of the dataset of Knauer et al. 2002 must be downloaded from the link: https://doi.org/10.6084/m9.figshare.19681410.v1 
A copy of the used version of the data set in our analyses has been included in the files (gm_dataset_Knauer_et_al_2022.xlsx). 

All the codes were written in Python and tested on version 3.11.3. 

The required packages and the tested version of them are as follows:

pandas 1.5.3

numpy 1.24.3

sklearn 1.2.2

scipy 1.10.1

itertools


The data set contains more than 100 structural, anatomical, biochemical, and physiological traits, each provided in a separate column. In the paper, we used ten anatomical and structural traits as follows:


```
list_of_traits = ['LMA','T_mesophyll','fias_mesophyll','T_cw','T_cyt','T_chloroplast','Sm','Sc','T_leaf', 'D_leaf']
```

The data set also was categorized into major plant functional types (PFTs) in the column 'plant_functional_type'. In our analyses, we used 13 PFTs from this column and five groups of them and a global set including all of the used PFTs. We identify each group or individual PFT by a list containing the names of the PFTs as follows:

```
PFTs = {'semi_deciduous_angiosperms':['semi-deciduous angiosperms'],
        'deciduous_gymnosperms':['deciduous gymnosperms'],
        'woody_evergreen_angiosperms':['evergreen angiosperms'],
        'C3_perennial_herbaceous':['C3 perennial herbaceous'],
        'ferns':['ferns'], 
        'C3_annual_herbaceous':['C3 annual herbaceous'],
        'C4_annual_herbaceous':['C4 annual herbaceous'],
        'C4_perennial_herbaceous':['C4 perennial herbaceous'],
        'evergreen_gymnosperms':['evergreen gymnosperms'],
        'CAM_plants':['CAM plants'],
        'mosses':['mosses'],
        'woody_deciduous_angiosperms':['deciduous angiosperms'],
        'fern_allies':['fern allies'],
        'C3_herbaceous':['C3 perennial herbaceous','C3 annual herbaceous'],
        'C3_C4_herbaceous':['C4 annual herbaceous','C4 perennial herbaceous','C3 perennial herbaceous','C3 annual herbaceous'],
        'woody_evergreens':['evergreen angiosperms','evergreen gymnosperms'],
        'woody_angiosperms':['evergreen angiosperms','deciduous angiosperms','semi-deciduous angiosperms'],
        'extended_ferns':['ferns', 'fern allies'],
        'global_set':['C3 perennial herbaceous','evergreen angiosperms','ferns','mosses','semi-deciduous angiosperms',
  'evergreen gymnosperms','C4 annual herbaceous','fern allies','deciduous gymnosperms',
  'deciduous angiosperms','CAM plants','C3 annual herbaceous','C4 perennial herbaceous']}
```
where the keys are arbitrary names for each group, and the values are the defined list for each PFT or group required in the functions.

Having the trait(s) and PFT(s) of interest in the mentioned format, all the results of the paper can be reproduced by using the appropriate function from the file "Leaf_gm_architecture_functions.py". 
Detailed explanations of the functions are included inside each of them.

All the results were achieved from aggregated data by the method explained in the paper. Hence, to get the same results, the data set first should be aggregated using the function: "data_aggregation()" before using each of the functions that need the data set.

For example, to get the predictability scores and Gini importance of the traits in the prediction scenario of cross-validation on the data of global set and combination of four traits, 'T_cw','Sc', ' T_leaf', and 'D_leaf' we use the following code in a Python file located in the same folder as the file "Leaf_gm_architecture_functions.py":

```
import pandas as pd
import Leaf_gm_architecture_functions as gm

global_df = pd.read_excel('gm_dataset_Knauer_et_al_2022.xlsx', sheet_name='data')
aggregated_df = gm.data_aggregation(global_df)
combination_of_traits = ['T_cw','Sc','T_leaf', 'D_leaf']

PFTs = {'semi_deciduous_angiosperms':['semi-deciduous angiosperms'],
        'deciduous_gymnosperms':['deciduous gymnosperms'],
        'woody_evergreen_angiosperms':['evergreen angiosperms'],
        'C3_perennial_herbaceous':['C3 perennial herbaceous'],
        'ferns':['ferns'], 
        'C3_annual_herbaceous':['C3 annual herbaceous'],
        'C4_annual_herbaceous':['C4 annual herbaceous'],
        'C4_perennial_herbaceous':['C4 perennial herbaceous'],
        'evergreen_gymnosperms':['evergreen gymnosperms'],
        'CAM_plants':['CAM plants'],
        'mosses':['mosses'],
        'woody_deciduous_angiosperms':['deciduous angiosperms'],
        'fern_allies':['fern allies'],
        'C3_herbaceous':['C3 perennial herbaceous','C3 annual herbaceous'],
        'C3_C4_herbaceous':['C4 annual herbaceous','C4 perennial herbaceous','C3 perennial herbaceous','C3 annual herbaceous'],
        'woody_evergreens':['evergreen angiosperms','evergreen gymnosperms'],
        'woody_angiosperms':['evergreen angiosperms','deciduous angiosperms','semi-deciduous angiosperms'],
        'extended_ferns':['ferns', 'fern allies'],
        'global_set':['C3 perennial herbaceous','evergreen angiosperms','ferns','mosses','semi-deciduous angiosperms',
  'evergreen gymnosperms','C4 annual herbaceous','fern allies','deciduous gymnosperms',
  'deciduous angiosperms','CAM plants','C3 annual herbaceous','C4 perennial herbaceous']}

results=gm.CV_with_PFT_and_combination_of_interest(aggregated_df,PFTs['global_set'],combination_of_traits,enseble_size=50,min_rows=50)

"""
The ensemble number must be increased to get more accurate results.
"""

print(results)
```

As another example, to obtain the predictability scores and total importance of the traits across all models with different combinations of the traits (based on the data availability) in the prediction scenario with the "ferns" as the test set and the rest of the PFTs as the train set, we use the following code in a Python file located in the same folder as the file "Leaf_gm_architecture_functions.py":

```
import pandas as pd
import Leaf_gm_architecture_functions as gm

global_df = pd.read_excel('gm_dataset_Knauer_et_al_2022.xlsx', sheet_name='data')
aggregated_df = gm.data_aggregation(global_df)
list_of_traits = ['LMA','T_mesophyll','fias_mesophyll','T_cw','T_cyt','T_chloroplast','Sm','Sc','T_leaf', 'D_leaf']

table_of_results=gm.cross_prediction_global_PFT(aggregated_df,['ferns'],list_of_traits,enseble_size=5,
                                             minimum_train_rows=40,minimum_test_rows=10)
"""
The ensemble number must be increased to get more accurate results.
The table_of_results contains the predictability scores and Gini importance of the traits in each trained model (different rows).
"""

if table_of_results.shape[0]>0:
    """Save the results as a csv file"""
    table_of_results.to_csv('table_of_results.csv', index=False)

    print('There are '+str(table_of_results.shape[0])+' trained models.')
else:
    print('There are no trained models because of data availability!')


average_imps_g,average_values_c=gm.total_importances(table_of_results) 

print("The IMP_G of the contributing traits in the trained models:")
print(average_imps_g)
print("The IMP_C of the contributing traits in the trained models:")
print(average_values_c)
```
