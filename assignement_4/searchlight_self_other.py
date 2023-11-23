# %%
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_all_trials.pkl', 'rb')
model1, lsa_dm, conditions_label, b_maps = pickle.load(f)
f.close()

# %%
# selecting self and other trials
now = datetime.now()
print('Renaming labels to S, O, and B:',now.strftime("%H:%M:%S"))

#from nilearn import datasets
from nilearn.image import new_img_like, load_img, index_img, clean_img, concat_imgs
from sklearn.model_selection import train_test_split, GroupKFold
n_trials=len(conditions_label)

#Concatenate beta maps
b_maps_conc=concat_imgs(b_maps)
#print(b_maps_conc.shape)
del b_maps
# Reshaping data------------------------------
from nilearn.image import index_img, concat_imgs

# ------------------------------------------------new
#Find all self and other trials
idx_neg=[int(i) for i in range(len(conditions_label)) if 'N' in conditions_label[i]]
idx_pos=[int(i) for i in range(len(conditions_label)) if 'P' in conditions_label[i]]
idx_but=[int(i) for i in range(len(conditions_label)) if 'B_' in conditions_label[i]]
identify_oth=[int(i) for i in range(len(conditions_label)) if 'O' in conditions_label[i]]
identify_self=[int(i) for i in range(len(conditions_label)) if 'S' in conditions_label[i]]

#correct order of trials
# Create individual arrays
zeros = np.zeros(90, dtype=int)
ones = np.ones(90, dtype=int)

# Concatenate arrays in the desired order
idx_oth = np.concatenate([zeros, ones, zeros, ones, zeros, ones]) # times six for all trials
idx_self = np.concatenate([ones, zeros, ones, zeros, ones, zeros])

#print(idx_neg)
#print(conditions_label)
# for i in range(len(conditions_label)): 
#     if i in idx_neg:
#         conditions_label[i]='N'
#     if i in idx_pos:
#         conditions_label[i]='P'
#     if i in idx_but:
#         conditions_label[i]='B'
# print(conditions_label)

#change to idx_other and self
for i in range(len(conditions_label)): 
    if idx_oth[i] == 1:
        conditions_label[i]='O'
    if idx_self[i] == 1:
        conditions_label[i]='S'
    if i in idx_but:
        conditions_label[i]='B'
print(conditions_label)

# -------------------------------- from here on, run the rest separately for each condition

now = datetime.now()
print('Selecting to S and O:',now.strftime("%H:%M:%S"))
# Make index of relevant trials
idx=np.concatenate((identify_self, identify_oth))  # change to idx_self, idx_oth
#print(idx)

#Select trials
conditions=np.array(conditions_label)[idx]
print(conditions)

#Select images
b_maps_img = index_img(b_maps_conc, idx)
print(b_maps_img.shape)

# %%
idx_oth[0]

# %%
# create training and testing vars on the basis of class labels

now = datetime.now()
print('Making a trial and test set:',now.strftime("%H:%M:%S"))
#conditions_img=conditions[idx]
#print(conditions_img)
#Make an index for spliting fMRI data with same size as class labels
idx2=np.arange(conditions.shape[0])

# create training and testing vars on the basis of class labels
idx_train,idx_test, conditions_train,  conditions_test = train_test_split(idx2,conditions, test_size=0.2)
#print(idx_train, idx_test)

# Reshaping data------------------------------
from nilearn.image import index_img
fmri_img_train = index_img(b_maps_img, idx_train)
fmri_img_test = index_img(b_maps_img, idx_test)
#Check data sizes
print('Trial and test set shape:')
print(fmri_img_train.shape)
print(fmri_img_test.shape)

# Saving the objects:
f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_testtrain_SO.pkl', 'wb')
pickle.dump([fmri_img_train, fmri_img_test, idx_train,idx_test, conditions_train,  conditions_test], f)
f.close()

now = datetime.now()
print('Trial and test set saved:',now.strftime("%H:%M:%S"))

# %%
# searchlight analysis

now = datetime.now()
print('Making a mask for analysis:',now.strftime("%H:%M:%S"))
# -------------------
import pandas as pd
import numpy as np
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import decoding
from nilearn.decoding import SearchLight
from sklearn import naive_bayes, model_selection #import GaussianNB

#########################################################################
#Make a mask with the whole brain

mask_wb_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0121/anat/sub-0121_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
anat_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0121/anat/sub-0121_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
#Load the whole brain mask
mask_img = load_img(mask_wb_filename)

## This bit can be used if you want to make a smaller select of the brain to study (e.g. to speed up analsysis)
# .astype() makes a copy.
process_mask = mask_img.get_fdata().astype(int)
#Set slices below x in the z-dimension to zero (in voxel space)
process_mask[..., :10] = 0
#Set slices above x in the z-dimension to zero (in voxel space)
process_mask[..., 170:] = 0
process_mask_img = new_img_like(mask_img, process_mask)

#Plot the mask on an anatomical background
plot_img(process_mask_img, bg_img=anat_filename,#bg_img=mean_fmri,
         title="Mask", display_mode="z",cut_coords=[-60,-50,-30,-10,10,30,50,70,80],
         vmin=.40, cmap='jet', threshold=0.9, black_bg=True)

# %%
now = datetime.now()
print('Starting searchlight analysis:',now.strftime("%H:%M:%S"))
#n_jobs=-1 means that all CPUs will be used

from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = SearchLight(
    mask_img,
    estimator=GaussianNB(),
    process_mask_img=process_mask_img,
    radius=5, n_jobs=-1,
    verbose=10, cv=10)
searchlight.fit(fmri_img_train, conditions_train)

now = datetime.now()
print('Finishing searchlight analysis:',now.strftime("%H:%M:%S"))

# %%
import pickle
import nilearn

#Save the searchlight model

# Saving the objects:
f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_all_trials_searchlight_SO.pkl', 'wb')
pickle.dump([searchlight, searchlight.scores_], f)
f.close()

# Getting back the objects:
#f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_all_trials_searchlight_SO.pkl', 'rb')
#searchlight,searchlight_scores_ = pickle.load(f)
#f.close()

# Getting back the objects:
#f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_testtrain_SO.pkl', 'rb')
#fmri_img_train, fmri_img_test, idx_train,idx_test, conditions_train,  conditions_test= pickle.load(f)
#f.close()

now = datetime.now()
print('Searchlight output saved:',now.strftime("%H:%M:%S"))

# %%
#plot results

from nilearn import image, plotting
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import new_img_like, load_img
mask_wb_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0121/anat/sub-0121_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
anat_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0121/anat/sub-0121_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

now = datetime.now()
print('Plotting and saving searchlight output (threshold:0.6):',now.strftime("%H:%M:%S"))

#Create an image of the searchlight scores
searchlight_img = new_img_like(anat_filename, searchlight.scores_)


plot_glass_brain(searchlight_img, cmap='jet',colorbar=True, threshold=0.5,
                          title='self vs other (unthresholded)',
                          plot_abs=False)

fig=plotting.plot_glass_brain(searchlight_img,cmap='prism',colorbar=True,threshold=0.60,title='self vs other (Acc>0.6')
fig.savefig("/work/sarah_a_folder/ass4/InSpe_neg_vs_but_searchlightNB_glass_SO.png", dpi=300)
#plt.show()

plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False,
              title='pos vs neg (Acc>0.6)')
plt.show()

# %%
print('Number of voxels in searchlight: ',searchlight.scores_.size)
#Find the percentile that makes the cutoff for the 500 best voxels
perc=100*(1-500.0/searchlight.scores_.size)
#Print percentile
print('Percentile for 500 most predictive voxels: ',perc)
#Find the cutoff
cut=np.percentile(searchlight.scores_,perc)
#Print cutoff
print('Cutoff for 500 most predictive voxels: ', cut)
#cut=0
#Make a mask using cutoff

#Load the whole brain mask
mask_img2 = load_img(mask_wb_filename)

# .astype() makes a copy.
process_mask2 = mask_img2.get_fdata().astype(int)
process_mask2[searchlight.scores_<=cut] = 0
process_mask2_img = new_img_like(mask_img2, process_mask2)

# %%
# --------------
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import plotting

#Create an image of the searchlight scores
searchlight_img = new_img_like(anat_filename, searchlight.scores_)
#Plot the searchlight scores on an anatomical background
plot_img(searchlight_img, bg_img=anat_filename,#bg_img=mean_fmri,
         title="Searchlight", display_mode="z",cut_coords=[-25,-20,-15,-10,-5,0,5],
         vmin=.40, cmap='jet', threshold=cut, black_bg=True)

#plotting.plot_glass_brain effects
fig=plotting.plot_glass_brain(searchlight_img,threshold=cut)
fig.savefig("/work/sarah_a_folder/ass4/InSpe_neg_vs_but_searchlightNB_glass_500_SO.png", dpi=300)

now = datetime.now()
print('Saving glass brain with 500 most predictive voxels:',now.strftime("%H:%M:%S"))

# %%
# permutations
now = datetime.now()
print('Perform permutation test on test set using 500 predictive voxels:',now.strftime("%H:%M:%S"))
from sklearn.naive_bayes import GaussianNB
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=process_mask2_img, standardize=False)

# We use masker to retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_img_test)

#Print size of matrix (images x voxels)
print(fmri_masked.shape)

from sklearn.model_selection import permutation_test_score
score_cv_test, scores_perm, pvalue= permutation_test_score(
    GaussianNB(), fmri_masked, conditions_test, cv=10, n_permutations=1000, # change number of permutations 
    n_jobs=-1, random_state=0, verbose=0, scoring=None)
print("Classification Accuracy: %s (pvalue : %s)" % (score_cv_test, pvalue))


# %%
import pickle

now = datetime.now()
print('Saving permutation scores:',now.strftime("%H:%M:%S"))
#Save the permutation scores

# Saving the objects:
f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_all_trials_searchlight_perms_SO.pkl', 'wb')
pickle.dump([score_cv_test, scores_perm, pvalue], f)
f.close()

# Getting back the objects:
#f = open('/work/sarah_a_folder/ass4/InSpe_first_level_models_all_trials_searchlight_perms_SO.pkl', 'rb')
#score_cv_test, scores_perm, pvalue = pickle.load(f)
#f.close()

# %%
now = datetime.now()
print('Plotting and saving permutation scores:',now.strftime("%H:%M:%S"))

import numpy as np

#How many classes
n_classes = np.unique(conditions_test).size

plt.hist(scores_perm, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
plt.plot(2 * [score_cv_test], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Chance level')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')

plt.savefig("/work/sarah_a_folder/ass4/InSpe_self_vs_other_one_sub_perm_SO.png", dpi=300)
plt.show()


