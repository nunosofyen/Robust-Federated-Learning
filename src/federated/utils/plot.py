import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.font_manager

font = matplotlib.font_manager.FontProperties(family='Times New Roman', style='normal')
csfont = {'fontname':'Times New Roman'}
x_axis = range(150, 310, 10)

# set the axises
ax = plt.gca()
ax.set_ylim([0, 100])
# Add title and axis names
plt.xlabel('rounds', **csfont)
plt.ylabel('UAR[%]', **csfont)
plt.title('On Original Data', **csfont)
plt.plot(x_axis, fl_org, color='black', linestyle='dashed', marker="+", label='fl')
plt.plot(x_axis, ad_org_fgsm, color = 'red', marker='3', label='ad_fgsm')
plt.plot(x_axis, ad_org_pgd, color='#6E750E', marker='o', label='ad_pgd')
plt.plot(x_axis, ad_org_df, color='#FFD700', marker='*', label='ad_df')
plt.legend(prop=font, loc='lower right')
plt.savefig('client/results_demos/1.png')
plt.close()

ax = plt.gca()
ax.set_ylim([0, 100])
plt.xlabel('rounds')
plt.ylabel('UAR[%]')
plt.title('On Randomized  Data')
plt.plot(x_axis, fl_ran, color='black', linestyle='dashed', label='fl')
plt.plot(x_axis, ad_ran_fgsm, color = 'red', marker='3', label='ad_fgsm')
plt.plot(x_axis, ad_ran_pgd, color='#6E750E', marker='o', label='ad_pgd')
plt.plot(x_axis, ad_ran_df, color='#FFD700', marker='*', label='ad_df')
plt.legend(loc='lower right')
plt.savefig('client/results_demos/2.png')
plt.close()

ax = plt.gca()
ax.set_ylim([0, 100])
plt.xlabel('rounds')
plt.ylabel('UAR[%]')
plt.title('On Adversarial Data')
plt.plot(x_axis, fl_aa_fgsm, color='red', linestyle='dashed',  label='fl_fgsm')
plt.plot(x_axis, fl_aa_pgd, color='#6E750E', marker='o', linestyle='dashed', label='fl_pgd')
plt.plot(x_axis, fl_aa_df, color='#FFD700', marker='>', linestyle='dashed', label='fl_df')
plt.plot(x_axis, ad_aa_fgsm, color = 'red', marker='3', label='ad_fgsm')
plt.plot(x_axis, ad_aa_pgd, color='#6E750E', marker='o', label='ad_pgd')
plt.plot(x_axis, ad_aa_df, color='#FFD700', marker='*', label='ad_df')
plt.legend()
plt.savefig('client/results_demos/3.png')
plt.close()

ax = plt.gca()
ax.set_ylim([0, 100])
plt.xlabel('rounds')
plt.ylabel('UAR[%]')
plt.title('On Randomized Adversarial Data')
plt.plot(x_axis, fl_raa_fgsm, color='red', linestyle='dashed',  label='fl_fgsm')
plt.plot(x_axis, fl_raa_pgd, color='#6E750E', marker='o', linestyle='dashed', label='fl_pgd')
plt.plot(x_axis, fl_raa_df, color='#FFD700', marker='>', linestyle='dashed', label='fl_df')
plt.plot(x_axis, ad_raa_fgsm, color = 'red', marker='3', label='ad_fgsm')
plt.plot(x_axis, ad_raa_pgd, color='#6E750E', marker='o', label='ad_pgd')
plt.plot(x_axis, ad_raa_df, color='#FFD700', marker='*', label='ad_df')
plt.legend()
plt.savefig('client/results_demos/4.png')
plt.close()
