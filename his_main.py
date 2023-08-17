# -*- coding: utf-8 -*-
"""
This code computes the experiments related to the manuscript:
    "A network score-based metric to optimize the quality assurance of automatic target segmentation." by Rodriguez Outeiral et al.

"""
#%%
import black
import gc
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pandas as pd
from scipy.stats.stats import pearsonr,skew
import seaborn as sns
from misc import *
import surface_distance


# path definition and initializing variables
folds=[0,1,2,3,4]
phase = 'test'
task = 517 #144: mesorectum ; 517: cervix gtv
path_to_save_excel = r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_{}\all_features_phiro\all_metrics_{}.xlsx'.format(task,phase)

#Path definitions. Change this accordingly to reproduce with different data.
if task == 517:
    path_to_test = r'F:\project_3\Roque\data\nnUNet_raw_data_base\nnUNet_raw_data\Task517_Cervix'
    path_to_test_images = path_to_test + '/imagesTs/'
    path_to_test_labels = path_to_test + '/labelsTs/'
    path_to_split_pkls = r'F:\project_4\am_surrogate/data/Task517_Cervix'
    path_to_results = r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_517'
    
elif task == 144:
    path_to_test = r'F:\project_4\am_surrogate\data\Task144_mps_2ndBenchmarkUnmatched'
    path_to_test_images = path_to_test + '/imagesTs/'
    path_to_test_labels = path_to_test + '/labelsTs/'
    path_to_split_pkls = r'F:\project_4\am_surrogate\data\Task144_mps_2ndBenchmarkUnmatched'
    path_to_results = r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_144'

#Dataframe where both the segmentation metrics and the metrics derived from the score map are store. Per patient and fold.
if phase=="val": #for the val set we need to compute the HIS for all hyperparameters.
    df_all_metrics = pd.DataFrame(
        columns=['PatID', 'Dice', 'HD', 'MSD', "SurfDice_1", "SurfDice_2","SurfDice_3","SurfDice_5", "Mean", 
                 "HIS_05","HIS_10","HIS_15","HIS_20","HIS_25","HIS_30","HIS_35",
                 "HIS_40","HIS_45","STD", "fold"])
    patients = list(pd.read_pickle(path_to_split_pkls+'/internal_val_set.pkl')['PatID'])

else:
    df_all_metrics = pd.DataFrame(
        columns=['PatID', 'Dice', 'HD', 'MSD', "SurfDice_2", "Mean", "HIS_30", "STD", "fold"])
    patients = list(pd.read_pickle(path_to_split_pkls+'/external_blind_test_set.pkl')['PatID'])


#Running in the 5 different folds to get a sense of variability of initialization
for fold in folds:
    # Different outputs and score maps per fold
    path_to_out = path_to_results + '/out_dir_{}_fold_{}/'.format(task,fold) # out
    path_to_am = path_to_results + '/AM_{}/'.format(fold) # AM_0, AM_1, AM_2, AM_3, AM_4
    path_to_seg_results = r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_{}\with_all_surfdsc_geometric_performance_fold_{}.csv'.format(task,fold)
    df_seg_results = pd.read_csv(path_to_seg_results,sep=';')
      
    for p,pname in enumerate(patients):
        if pname == 'CERVIX_456': 
            print('Her uterus was removed')
            continue
        img_nii = nib.load(path_to_test_images+pname+'_0000.nii.gz') # this one remains as nii to get the voxel sizes
        gt_nii = nib.load(path_to_test_labels+pname+'.nii.gz').get_fdata()
        am_nii = nib.load(path_to_am+pname+'.nii.gz').get_fdata()/255 #normalizing back to softmax
        un_nii = -(am_nii * np.log(am_nii) + (1-am_nii)*np.log(1-am_nii)) * 1/np.log(2)
        un_nii[np.isnan(un_nii)]=0
        out_nii = nib.load(path_to_out+pname+'.nii.gz').get_fdata()
        this_hd = df_seg_results[df_seg_results['PatID']==pname]['HD'].item()
        this_msd = df_seg_results[df_seg_results['PatID']==pname]['MSD'].item()
        if (this_hd != np.inf) and (this_msd != np.inf) and (this_hd != np.nan) and (this_msd != np.nan):

            this_dice=df_seg_results[df_seg_results['PatID']==pname]['Dice'].item()
            this_surf1=df_seg_results[df_seg_results['PatID']==pname]['surface_dice_coefficient_1.0'].item()
            this_surf2=df_seg_results[df_seg_results['PatID']==pname]['surface_dice_coefficient_2.0'].item()
            this_surf3=df_seg_results[df_seg_results['PatID']==pname]['surface_dice_coefficient_3.0'].item()
            this_surf5=df_seg_results[df_seg_results['PatID']==pname]['surface_dice_coefficient_5.0'].item()
            
        df_all_metrics = df_all_metrics.append({'PatID': pname, 
                             'Dice': this_dice,
                             'HD': this_hd,
                             'MSD': this_msd,
                             "SurfDice_1" : this_surf1,
                             "SurfDice_2" : this_surf2,
                             "SurfDice_3" : this_surf3,
                             "SurfDice_5" : this_surf5,
                             'Mean': am_nii[am_nii>0].mean(),
                             'HIS_05': am_nii[am_nii>0.05].mean(),
                             'HIS_10': am_nii[am_nii>0.10].mean(),
                             'HIS_15': am_nii[am_nii>0.15].mean(),
                             'HIS_20': am_nii[am_nii>0.20].mean(),
                             'HIS_25': am_nii[am_nii>0.25].mean(),
                             'HIS_30': am_nii[am_nii>0.30].mean(),
                             'HIS_35': am_nii[am_nii>0.35].mean(),
                             'HIS_40': am_nii[am_nii>0.40].mean(),
                             'HIS_45': am_nii[am_nii>0.45].mean(),
                             'Mean_un': un_nii[un_nii>0].mean(),
                             'STD': np.std(am_nii[am_nii>0]),
                             'fold': fold}, ignore_index=True)

df_all_metrics.to_excel(path_to_save_excel)

#%% Getting results
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
phase = 'test'
task=144
final_df = pd.read_excel(r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_{}\all_features_phiro\all_metrics_{}.xlsx'.format(task,phase))

metrics_score = ["Mean","HIS_05","HIS_10","HIS_15","HIS_20",
                 "HIS_25","HIS_30","HIS_35", "HIS_40","HIS_45","STD", "Mean_un"]
folds = [0,1,2,3,4]
metrics_seg = ['Dice','HD','MSD','SurfDice_5']

def get_result(met_score,met_seg):
    folds = [0,1,2,3,4]
    list_of_correlations = []
    for fold in folds:
        list_of_correlations.append(np.round(spearmanr(final_df[final_df['fold']==fold][met_seg],final_df[final_df['fold']==fold][met_score])[0],3))
    this_mean = np.mean(list_of_correlations)
    this_std = np.std(list_of_correlations)
    return this_mean, this_std, list_of_correlations

for met_sg in metrics_seg:
    for met_sc in metrics_score:
        print(met_sg, met_sc, get_result(met_sc, met_sg))
        

#%% Plots of the hyperparameter per fold (Figure 1)
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr

matplotlib.rcParams.update({'font.size': 40})
task=517
phase='val'#this analysis is always in the validation set.
final_df = pd.read_excel(r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_{}\all_features_phiro\all_metrics_{}.xlsx'.format(task,phase))
folds = [0,1,2,3,4]

def get_result_diff_per_fold(feature,metric):
    folds = [0,1,2,3,4]
    list_of_correlations = []
    for fold in folds:
        list_of_correlations.append(np.round(pearsonr(final_df[final_df['fold']==fold][metric],final_df[final_df['fold']==fold][feature])[0],3)
                                    -np.round(pearsonr(final_df[final_df['fold']==fold][metric],final_df[final_df['fold']==fold]['Mean'])[0],3))
    this_mean = np.mean(list_of_correlations)
    this_std = np.std(list_of_correlations)
    return this_mean, this_std, list_of_correlations

plt.figure(figsize=(12,12))
plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
         [get_result_diff_per_fold('Mean','Dice')[0],
          get_result_diff_per_fold('HIS_05','Dice')[0],get_result_diff_per_fold('HIS_10','Dice')[0],
          get_result_diff_per_fold('HIS_15','Dice')[0],get_result_diff_per_fold('HIS_20','Dice')[0],
          get_result_diff_per_fold('HIS_25','Dice')[0],get_result_diff_per_fold('HIS_30','Dice')[0],
          get_result_diff_per_fold('HIS_35','Dice')[0],get_result_diff_per_fold('HIS_40','Dice')[0],
          get_result_diff_per_fold('HIS_45','Dice')[0]], linewidth=3, color='k')
for fold in folds:
    plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
             [get_result_diff_per_fold('Mean','Dice')[2][fold],
              get_result_diff_per_fold('HIS_05','Dice')[2][fold],get_result_diff_per_fold('HIS_10','Dice')[2][fold],
              get_result_diff_per_fold('HIS_15','Dice')[2][fold],get_result_diff_per_fold('HIS_20','Dice')[2][fold],
              get_result_diff_per_fold('HIS_25','Dice')[2][fold],get_result_diff_per_fold('HIS_30','Dice')[2][fold],
              get_result_diff_per_fold('HIS_35','Dice')[2][fold],get_result_diff_per_fold('HIS_40','Dice')[2][fold],
              get_result_diff_per_fold('HIS_45','Dice')[2][fold]], linewidth=2, linestyle='--', color='0.5')

plt.ylim(-0.3,0.3)
plt.axhline(y=0, color='r', linestyle='-')

plt.xlabel('λ')
plt.ylabel('Δr (r HiS - r Mean)')
plt.title('DSC')
plt.tight_layout()

plt.figure(figsize=(12,12))
plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
         [get_result_diff_per_fold('Mean','HD')[0],
          get_result_diff_per_fold('HIS_05','HD')[0],get_result_diff_per_fold('HIS_10','HD')[0],
          get_result_diff_per_fold('HIS_15','HD')[0],get_result_diff_per_fold('HIS_20','HD')[0],
          get_result_diff_per_fold('HIS_25','HD')[0],get_result_diff_per_fold('HIS_30','HD')[0],
          get_result_diff_per_fold('HIS_35','HD')[0],get_result_diff_per_fold('HIS_40','HD')[0],
          get_result_diff_per_fold('HIS_45','HD')[0]], linewidth=3, color='k')
for fold in folds:
    plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
             [get_result_diff_per_fold('Mean','HD')[2][fold],
              get_result_diff_per_fold('HIS_05','HD')[2][fold],get_result_diff_per_fold('HIS_10','HD')[2][fold],
              get_result_diff_per_fold('HIS_15','HD')[2][fold],get_result_diff_per_fold('HIS_20','HD')[2][fold],
              get_result_diff_per_fold('HIS_25','HD')[2][fold],get_result_diff_per_fold('HIS_30','HD')[2][fold],
              get_result_diff_per_fold('HIS_35','HD')[2][fold],get_result_diff_per_fold('HIS_40','HD')[2][fold],
              get_result_diff_per_fold('HIS_45','HD')[2][fold]], linewidth=2, linestyle='--', color='0.5')
plt.axhline(y=0, color='r', linestyle='-')

plt.xlabel('λ')
plt.ylabel('Δr (r HiS - r Mean)')
plt.title('95th HD')
plt.tight_layout()

plt.figure(figsize=(12,12))
plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
         [get_result_diff_per_fold('Mean','MSD')[0],
          get_result_diff_per_fold('HIS_05','MSD')[0],get_result_diff_per_fold('HIS_10','MSD')[0],
          get_result_diff_per_fold('HIS_15','MSD')[0],get_result_diff_per_fold('HIS_20','MSD')[0],
          get_result_diff_per_fold('HIS_25','MSD')[0],get_result_diff_per_fold('HIS_30','MSD')[0],
          get_result_diff_per_fold('HIS_35','MSD')[0],get_result_diff_per_fold('HIS_40','MSD')[0],
          get_result_diff_per_fold('HIS_45','MSD')[0]], linewidth=3, color='k')
for fold in folds:
    plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
             [get_result_diff_per_fold('Mean','MSD')[2][fold],
              get_result_diff_per_fold('HIS_05','MSD')[2][fold],get_result_diff_per_fold('HIS_10','MSD')[2][fold],
              get_result_diff_per_fold('HIS_15','MSD')[2][fold],get_result_diff_per_fold('HIS_20','MSD')[2][fold],
              get_result_diff_per_fold('HIS_25','MSD')[2][fold],get_result_diff_per_fold('HIS_30','MSD')[2][fold],
              get_result_diff_per_fold('HIS_35','MSD')[2][fold],get_result_diff_per_fold('HIS_40','MSD')[2][fold],
              get_result_diff_per_fold('HIS_45','MSD')[2][fold]], linewidth=2, linestyle='--', color='0.5')

plt.ylim(-0.3,0.3)
plt.axhline(y=0, color='r', linestyle='-')

plt.xlabel('λ')
plt.ylabel('Δr (r HiS - r Mean)')
plt.title('MSD')

plt.tight_layout()

plt.figure(figsize=(12,12))
plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
         [get_result_diff_per_fold('Mean','SurfDice_5')[0],
          get_result_diff_per_fold('HIS_05','SurfDice_5')[0],get_result_diff_per_fold('HIS_10','SurfDice_5')[0],
          get_result_diff_per_fold('HIS_15','SurfDice_5')[0],get_result_diff_per_fold('HIS_20','SurfDice_5')[0],
          get_result_diff_per_fold('HIS_25','SurfDice_5')[0],get_result_diff_per_fold('HIS_30','SurfDice_5')[0],
          get_result_diff_per_fold('HIS_35','SurfDice_5')[0],get_result_diff_per_fold('HIS_40','SurfDice_5')[0],
          get_result_diff_per_fold('HIS_45','SurfDice_5')[0]], linewidth=3, color='k')
for fold in folds:
    plt.plot( [0,0.05,0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
             [get_result_diff_per_fold('Mean','SurfDice_5')[2][fold],
              get_result_diff_per_fold('HIS_05','SurfDice_5')[2][fold],get_result_diff_per_fold('HIS_10','SurfDice_5')[2][fold],
              get_result_diff_per_fold('HIS_15','SurfDice_5')[2][fold],get_result_diff_per_fold('HIS_20','SurfDice_5')[2][fold],
              get_result_diff_per_fold('HIS_25','SurfDice_5')[2][fold],get_result_diff_per_fold('HIS_30','SurfDice_5')[2][fold],
              get_result_diff_per_fold('HIS_35','SurfDice_5')[2][fold],get_result_diff_per_fold('HIS_40','SurfDice_5')[2][fold],
              get_result_diff_per_fold('HIS_45','SurfDice_5')[2][fold]], linewidth=2, linestyle='--', color='0.5')

plt.ylim(-0.3,0.3)
plt.axhline(y=0, color='r', linestyle='-')

plt.xlabel('λ')
plt.ylabel('Δr (r HiS - r Mean)')
plt.title('Surface DSC')
plt.tight_layout()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
#Code for AUC curves 
import os
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
list_of_aucs = {0:{},1:{},2:{},3:{},4:{}} # one per fold}
folds = [0,1,2,3,4]
features = ['Mean', 'HiS', 'STD', 'Mean (entropy)']#, 'RiF'] #, 'RIF', 'SIRF'
task = 517 #517 (brachy cervix GTV) or 144 (meso CTV)
phase='test' #should be test. We have val for now to not make any decisions on the test set
metric = 'MSD'
path_to_experiments = r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_{}\all_features_phiro\all_metrics_{}.xlsx'.format(task,phase)
df_corr = pd.read_excel(path_to_experiments)

for fold in folds:
    for feature in features:
        list_of_aucs[fold][feature]=[]
        these_msds = df_corr[df_corr['fold']==fold]['MSD']
        these_feats = df_corr[df_corr['fold']==fold][feat]
        df_corr = pd.read_excel(path_to_experiments)
        if feature == 'HiS': 
            feat = 'Mean_0_{}'.format(best_hic) #change depending on task
        elif feature =='STD':
            feat = 'STD'
        elif feature =='Mean (entropy)':
            feat = 'Mean'
        else: 
            feat = feature
        these_feats = df_corr[feat]
        if metric=='Dice':
            thr_metrics = [0.0005 * (x+1) + 0.0005 for x in range(2000)] #thresholds to define the ROC curves. Each threshold of MSD, one ROC curve
        else:# if its not positively correlated, it should be "less than"
            thr_metrics = [0.05 * (x+1) + 0.05 for x in range(2000)] #thresholds to define the ROC curves. Each threshold of MSD, one ROC curve

        
        for this_thr in thr_metrics:
            if metric=='Dice' or (feat =='STD' and task == 144) or (feature =='Mean (entropy)' and task == 144):
                # print('you are here')
                this_fpr,this_tpr,thr = metrics.roc_curve(np.array(these_msds>this_thr), these_feats,drop_intermediate=False) 
            else:# if its not positively correlated, it should be "less than"
                this_fpr,this_tpr,thr = metrics.roc_curve(np.array(these_msds<this_thr), these_feats,drop_intermediate=False) 

            this_auc = metrics.auc(this_fpr, this_tpr)
            list_of_aucs[fold][feature].append(this_auc)
#plotting
# computing mean over folds:
mean_auc = {'Mean':[], 'HiS':[], 'STD':[], 'Mean (entropy)':[]}
std_auc = {'Mean':[], 'HiS':[], 'STD':[], 'Mean (entropy)':[]}
for feature in features:
    for p_auc in range(2000):
        this_point_per_fold=[]
        for fold in folds:
            this_point_per_fold.append(list_of_aucs[fold][feature][p_auc])
        mean_folds = np.mean(this_point_per_fold)
        std_folds = np.std(this_point_per_fold)
        mean_auc[feature].append(mean_folds)
        std_auc[feature].append(std_folds)
    
plt.figure(figsize=(12,12))
matplotlib.rcParams.update({'font.size': 30})
plt.style.use('seaborn-dark-palette')
for feature in features:
    plt.plot(thr_metrics,mean_auc[feature],  linewidth=3, label=feature)
    if metric == 'Dice':
        plt.xlabel(metric)
    elif metric == 'HD':
        plt.xlabel('95th HD (mm)')
    elif metric == 'MSD':
        plt.xlabel('MSD (mm)')
    plt.ylabel('AUC')
    plt.legend(loc='lower right', ncol=2)
plt.ylim(0,1)
if task == 517: plt.title('Cervical cancer task')
if task == 144: plt.title('Rectal cancer task')
if metric == 'HD': plt.xlim(2,23)
if metric == 'MSD': plt.xlim(0,7)
plt.tight_layout()
# plt.savefig(r'F:\important_documents\My_papers\Paper 4\Journal\before_submitting\figures\Figure_4\{}_{}_dpi.eps'.format(metric,task),dpi=2400)

#%% Plots 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
import os
import matplotlib

task = 517 #517 (brachy cervix GTV) or 144 (meso CTV)
phase='test'
fold=3 #fold 3 for 517. Fold 2 por 144.
final_df = pd.read_excel(r'F:\project_4\am_surrogate\experiments\histogram_based_features\dataset_{}\all_features_phiro\all_metrics_{}.xlsx'.format(task,phase))

plt.style.use('seaborn-dark-palette')
matplotlib.rcParams.update({'font.size': 40})

def plot_correlation(list_feature, list_metric, thr_metric = 0, name_feat='HiC', name_metric='MSD (mm)', list_names=''):
    plt.figure(figsize=(12,12))
    sns.regplot(x=list_feature, y=list_metric, fit_reg=True);
    plt.scatter(np.array(list_feature)[(np.array(list_metric)<=thr_metric)], np.array(list_metric)[(np.array(list_metric)<=thr_metric)])
    plt.scatter(np.array(list_feature)[(np.array(list_metric)<=thr_metric)], np.array(list_metric)[(np.array(list_metric)<=thr_metric)])
    if list_names: #they need to be ordered
        for n,name in enumerate(list_names):
            x, y = list_feature[n], list_metric[n]
            plt.text(x+.003, y+.003, name, fontsize=9)
    this_pearson_r = np.round(pearsonr(list_metric,list_feature)[0],3)
    this_pv = pearsonr(list_metric,list_feature)[1]
    plt.title('R = {}'.format(this_pearson_r))
    if 'DSC' not in name_metric: 
        plt.ylabel(name_metric + ' (mm)')
    else:
        plt.ylabel(name_metric)
    plt.xlabel(name_feat)
    plt.tight_layout()
    return this_pv



# Correlation plots to reproduce figure 
# The lists are per fold as this dataframe has all the folds together
list_of_hiss = list(final_df[final_df['fold']==fold]['HIS_30'])
list_of_msds = list(final_df[final_df['fold']==fold]['MSD'])
list_of_hds = list(final_df[final_df['fold']==fold]['HD'])
list_of_dices = list(final_df[final_df['fold']==fold]['Dice'])
list_of_surfs = list(final_df[final_df['fold']==fold]['SurfDice_5'])
plot_correlation(list_of_hiss, list_of_dices, 0, name_metric='DSC',name_feat='HiS (λ=0.30)',list_names='')
plot_correlation(list_of_hiss, list_of_hds, 0, name_metric='95th HD',name_feat='HiS (λ=0.30)',list_names='')
plot_correlation(list_of_hiss, list_of_msds, 0, name_metric='MSD',name_feat='HiS (λ=0.30)',list_names='')
plot_correlation(list_of_hiss, list_of_surfs, 0, name_metric='Surface DSC',name_feat='HiS (λ=0.30)',list_names='')

# Residual plots to check for linearity

plt.figure(figsize=(12,12))
sns.residplot(data=final_df[final_df['fold']==fold], x='HIS_30', y='Dice')
plt.title('')
plt.xlabel('HiS (λ=0.30)')
plt.ylabel('Δ DSC')
plt.tight_layout()

plt.figure(figsize=(12,12))
sns.residplot(data=final_df[final_df['fold']==fold], x='HIS_30', y='HD')
plt.title('')
plt.xlabel('HiS (λ=0.30)')
plt.ylabel('Δ 95th HD (mm)')
plt.tight_layout()

plt.figure(figsize=(12,12))
sns.residplot(data=final_df[final_df['fold']==fold], x='HIS_30', y='MSD')
plt.title('')
plt.xlabel('HiS (λ=0.30)')
plt.ylabel('Δ MSD (mm)')
plt.tight_layout()

plt.figure(figsize=(12,12))
sns.residplot(data=final_df[final_df['fold']==fold], x='HIS_30', y='SurfDice_5')
plt.title('')
plt.xlabel('HiS (λ=0.30)')
plt.ylabel('Δ Surface DSC')
plt.tight_layout()
