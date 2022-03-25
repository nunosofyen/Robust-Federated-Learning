import numpy as np
from numpy import genfromtxt
from utils.score import scoring
import pandas as pd
import csv
import os
csv_path = 'client/results/out_csv/'
truth = genfromtxt(os.path.join(csv_path, 'truth.csv'), delimiter=",")

models = ['dynamic-vgg16-180rounds-68clients-1epochs']
advs = ['fgsm']
for model in models:
    org = genfromtxt(os.path.join(csv_path, model, 'org.csv'), delimiter=",")
    for adv in advs:
        adv_pre = genfromtxt(os.path.join(csv_path, model, '{}.csv'.format(adv)), delimiter=",")
        ran_org_0 = genfromtxt(os.path.join(csv_path, model, '{}_ran_org_{}.csv'.format(adv, str(0))), delimiter=",")
        ran_org_1 = genfromtxt(os.path.join(csv_path, model, '{}_ran_org_{}.csv'.format(adv, str(1))), delimiter=",")
        ran_org_2 = genfromtxt(os.path.join(csv_path, model, '{}_ran_org_{}.csv'.format(adv, str(2))), delimiter=",")
        ran_adv_0 = genfromtxt(os.path.join(csv_path, model, '{}_ran_adv_{}.csv'.format(adv, str(0))), delimiter=",")
        ran_adv_1 = genfromtxt(os.path.join(csv_path, model, '{}_ran_adv_{}.csv'.format(adv, str(1))), delimiter=",")
        ran_adv_2 = genfromtxt(os.path.join(csv_path, model, '{}_ran_adv_{}.csv'.format(adv, str(2))), delimiter=",")

        cm_org, uar_org = scoring(truth, org)
        cm_adv, uar_adv = scoring(truth, adv_pre)
        cm_ran_org_0, uar_ran_org_0 = scoring(truth, ran_org_0)
        cm_ran_org_1, uar_ran_org_1 = scoring(truth, ran_org_1)
        cm_ran_org_2, uar_ran_org_2 = scoring(truth, ran_org_2)
        cm_ran_adv_0, uar_ran_adv_0 = scoring(truth, ran_adv_0)
        cm_ran_adv_1, uar_ran_adv_1 = scoring(truth, ran_adv_1)
        cm_ran_adv_2, uar_ran_adv_2 = scoring(truth, ran_adv_2)
        cm_ran_org = np.mean(np.array([cm_ran_org_0, cm_ran_org_1, cm_ran_org_2]), axis=0, dtype=int)
        cm_ran_adv = np.mean(np.array([cm_ran_adv_0, cm_ran_adv_1, cm_ran_adv_2]), axis=0, dtype=int)
        uar_ran_org = uar_ran_org_0 / 3 + uar_ran_org_1 / 3  + uar_ran_org_2 / 3
        uar_ran_adv = uar_ran_adv_0 / 3 + uar_ran_adv_1 / 3  + uar_ran_adv_2 / 3
        print(model, adv)
        print('org uar:{:.4f}'.format(uar_org))
        print('adv uar:{:.4f}'.format(uar_adv))
        print('ran org uar:{:.4f}'.format(uar_ran_org))
        print('ran adv uar:{:.4f}'.format(uar_ran_adv))
