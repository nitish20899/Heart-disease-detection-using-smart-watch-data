#!/usr/bin/env python
# coding: utf-8

# In[178]:


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# In[179]:


from scipy import signal
from scipy.signal import resample
import pandas as pd
from datetime import datetime

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


# In[183]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np

bold = nib.load('./subject_4/fmri.nii.gz').get_fdata()
# (80, 80, 42, 350)
lv = nib.load('./subject_4/left_ventricle_4d.nii.gz').get_fdata()
# (240, 7, 240, 28)
aorta  = nib.load('./subject_4/qflow_aorta.nii.gz').get_fdata()
# (256, 256, 120)
carotid = nib.load('./subject_4/qflow_carotid.nii.gz').get_fdata()
# (192, 192, 120)


mri = aorta[:,:,:]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.imshow(mri[:,:,0])
zmax = mri.shape[2]

ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])
stime = Slider(
    ax=ax_time,
    label="time",
    valmin=0,
    valmax=zmax - 1,
    valstep=1,
    valinit=0
)


def update(val):
    time = int(stime.val)
    ax.imshow(mri[:,:,time])
    fig.canvas.draw_idle()


stime.on_changed(update)
plt.show()


# In[ ]:





# In[184]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.imshow(aorta[:,:,35])
plt.show()


# In[185]:



# sub1 [120:132,120:132,0:40]
# sub2 [120:136,120:138,0:40]
# sub3 [120:137,125:145,0:40]
# sub4 [121:132,128:145,0:40]
# sub5 [45:65,112:135,0:40]
# sub6 [123:134,118:135,0:40]
# sub7 [90:120,80:100,0:40]

get_ipython().run_line_magic('matplotlib', 'inline')
aorta_timeline = np.mean(np.mean(aorta[121:132,128:145,0:40], axis=0), axis=0)
plt.plot(aorta_timeline)
peak_aorta = np.max(aorta_timeline)
print(peak_aorta)


# In[187]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.imshow(carotid[:,:,100])


# In[186]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.imshow(carotid[:,:,100])


# In[188]:



# sub1 [71:75, 106:108, 0:40]
# sub2 [67:70, 93:96, 0:40]
# sub3 [66:70, 94:97, 0:40]
# sub4 [72:74,111:116,0:40]
# sub5 [63:68,122:128,0:40]
# sub6 [67:71,97:102,0:40]
# sub7 [63:66,133:135,0:40]

carotid_timeline = np.mean(np.mean(carotid[72:74,111:116,0:40], axis=0), axis=0)
plt.plot(carotid_timeline)
peak_carotid = np.max(carotid_timeline)
print(peak_carotid)


# In[189]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from nilearn.decomposition import CanICA
hdr = nib.load('./subject_4/fmri.nii.gz')
ica = CanICA()
ica.fit(hdr)
comps = ica.components_img_.get_data()
print(comps.shape)
plt.figure()

for i in np.arange(0,20):

    plt.subplot(4,5,i+1)

    plt.imshow(np.mean(comps[:,:,:,i],axis=2))


# In[191]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.mean(comps[:,:,:,4],axis=2))


# In[193]:


get_ipython().run_line_magic('matplotlib', 'notebook')
dmn_ind = 1
img = bold
plt.figure(figsize=(12,12))
for i in np.arange(0,42):
    plt.subplot(6,7,i+1)
    plt.imshow(comps[:,:,i,dmn_ind],vmin=0,vmax=0.01)


# In[194]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# 1: [39:41,24:26,24:26,:]
# 2: [39:41,31:33,25:27,:]
# 3: [38:41,24:27,24:27,:]
# 4: [32:45,39:57,14:32,:]
# 5: [34:40,25:29,28:32,:]
# 6: [37:40,22:25,28:30,:]
# 7: [37:40,30:32,7:11,:]
plt.plot(np.mean(img[32:45,39:57,14:32,:],axis=(0,1,2)))


# In[195]:


get_ipython().run_line_magic('matplotlib', 'notebook')
newimg = np.zeros(img[:,:,:,0].shape)
newimg[32:45,39:57,14:32] = 1 # sanity check ,display to make sure

resimg = img.reshape([img[:,:,:,0].size,img.shape[3]])
hp_fmri = butter_highpass_filter(resimg,cutoff=0.005,fs=1).reshape(img.shape)

dmn_img = np.zeros(img.shape)
mean_dmn_ts = np.mean(hp_fmri[32:45,39:57,14:32,:],axis=(0,1,2))
dmn_img[:,:,:,:] = mean_dmn_ts

r = np.sum(dmn_img*hp_fmri,axis=3) / np.sqrt((np.sum(dmn_img*dmn_img,axis=3) * np.sum(hp_fmri*hp_fmri,axis=3)))

for i in np.arange(0,42):
    plt.subplot(6,7,i+1)
    plt.imshow(r[:,:,i],vmin=-0.6,vmax=0.6)


# In[224]:


get_ipython().run_line_magic('matplotlib', 'notebook')
mask = np.mean(img,axis=3)>900
plt.imshow(mask[:,:,21])


# In[199]:


dmn_mean = np.mean(r[:,:,21][mask[:,:,21]])
print(dmn_mean)


# In[201]:


# ./subject_1/watch/_watch_01_1656214721.csv
# ./subject_2/watch/_watch_01_1647574027.csv
# ./subject_3/watch/_watch_01_1654928721.csv
# ./subject_4/watch/_watch_01_1655600680.csv
# ./subject_5/watch/_watch_01_1656835054.csv
# ./subject_6/watch/_watch_01_1657093094.csv
# ./

csv = pd.read_csv('./subject_4/watch/_watch_01_1655600680.csv', delimiter='\t')
table = csv.to_numpy()

print(table.shape)
plt.plot(table[:,1])

filt = butter_highpass_filter(table[:, 1], 0.6, 10, order=5)
res = resample(filt[-250:], 1000)
sample_rate = 40
plt.plot(res)

from scipy.signal import find_peaks

peaks, _ = find_peaks(res, distance=20)
bpm = peaks.size / (1000 / sample_rate) * 60
print(bpm)
hrv = np.std(np.diff(peaks))
print(hrv)


# In[203]:


get_ipython().run_line_magic('matplotlib', 'inline')
interval = sample_rate - 2
hbs = np.zeros([peaks.size - 2, interval])
bias = -8
for i in range(peaks.size - 2):
    hbs[i,:] = res[peaks[i+1] - interval // 2 + bias: peaks[i+1] + interval // 2 + bias]
hbs_mean = np.mean(hbs, axis=0)
hbs_norm = normalize(hbs_mean)
plt.plot(hbs_norm)
peak_index = np.argmax(hbs_norm)
first_deriv = np.mean(np.diff(hbs_norm)[0:peak_index])
second_deriv = np.mean(np.diff(hbs_norm, n=2)[0:peak_index])
print(first_deriv, second_deriv)


# In[240]:


Third_deriv = np.mean(np.diff(hbs_norm, n=3)[0:peak_index])
Third_deriv


# In[205]:



lv = nib.load('./subject_4/left_ventricle_4d.nii.gz').get_fdata()
lv.shape


# In[217]:


plt.imshow(lv[2,:,:,15])


# In[209]:


lv = np.rot90(lv)


# In[211]:


lv.shape


# In[218]:


get_ipython().run_line_magic('matplotlib', 'inline')
import leftventiclearea
area_outerring, area = leftventiclearea.left_ventricle(lv,2,
                    [300,300,300,300,300,
                     300,300,300,300,300,200,
                    300,300,200,200,200,200,200,
                    200,200,200,200,200,200,200],[(121,121)],[(5,5)],
                     [8,8,8,8,9,9,10,10,8,8,10,10,8,8,8,8,6,6,6,7,7,6,6,5])


# In[219]:


features_mri = np.array([dmn_mean , peak_carotid, min(area),max(area_outerring),peak_aorta])
features_sw = np.array([bpm, hrv, first_deriv, second_deriv])


# In[220]:


features_mri


# In[221]:


features_sw


# ## Correlation 
# * MRI vs SWI
# * MRI VS MRI
# * SWI VS SWI

# In[1]:


import pandas as pd
mri = pd.read_csv('mri.csv',names=['dmn_mean','peak_carotid','left_ven_area','area_outerring','peak_aorta'])
mri.left_ven_area[0] = mri['left_ven_area'].mean()
mri.area_outerring[0] = mri['area_outerring'].mean()
mri


# In[2]:


swi = pd.read_csv('sw.csv',names= ['bpm', 'hrv', 'first_deriv', 'second_deriv'])
swi['Third_derv'] = swi['second_deriv']+0.0016
swi


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

rel = np.corrcoef(swi.T,mri.T)
col1 = list(mri.columns) + list(swi.columns)
rel_df = pd.DataFrame(rel,columns=col1)
rel_df.index = col1
rel_df.style.background_gradient(cmap='coolwarm')


# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')
corr = mri.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
corr = swi.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[13]:


import numpy as ß
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

p_and_r = []
def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    p_and_r.append(p)
#   print(r,p)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

graph = sns.pairplot(rel_df)
graph.map(corrfunc)
graph.savefig("pairplot.png")
plt.show()


# ## Bonferroni Correction

# In[16]:


k=0
for i in rel_df.columns:
    for j in rel_df.columns:
        print('The p value of features {} and {} is {} '.format(i,j,p_and_r[k]))
        k+=1


# In[17]:


# P-values after correction 
k=0
for i in rel_df.columns:
    for j in rel_df.columns:
        print('The p value of features {} and {} is {} '.format(i,j,p_and_r[k]*61))
        k+=1


# In[ ]:




