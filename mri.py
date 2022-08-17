import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider

bold = nib.load('.\\subject_3\\fmri.nii.gz').get_fdata()
# (80, 80, 42, 350)
lv = nib.load('.\\subject_3\\left_ventricle_4d.nii.gz').get_fdata()
# (240, 7, 240, 28)
aorta  = nib.load('.\\subject_3\\qflow_aorta.nii.gz').get_fdata()
# (256, 256, 120)
carotid = nib.load('.\\subject_3\\qflow_carotid.nii.gz').get_fdata()
# (192, 192, 120)


mri = carotid[:,:,:]

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
