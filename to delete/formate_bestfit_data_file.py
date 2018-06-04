import os
import numpy as np
"""
this script is used to formate the best fit data in a way the error bars could be displayed properly
"""


#table file exported from GenX
par_file='D:\\table_processed.tab'
#best fit model files returned by the function domain_creator.print_data_for_publication2()
best_fit_file='D://model.xyz'
f0=open(par_file,'r')
f1=open(best_fit_file,'r')
f2=open(best_fit_file.replace('model','model_formated'),'w')
lines_f1=f1.readlines()
lines_f0=f0.readlines()
for line in lines_f1:
    items=line.rsplit('\t')
    id_components=items[0][0:-1].rsplit('_')
    N_dxdydz=0
    for line_f0 in lines_f0:
        items_f0=line_f0.rstrip().rsplit('\t')
        if 'gp' in items_f0[0]:
            checks=items_f0[0].rsplit('_')
            id_components_check=id_components[0][0:-1]+id_components[1]
            if float(id_components[1])<10:
                id_components_check=id_components[0][0:-1]+id_components[1]+id_components[0][0:-1]
                id_components_check2=id_components[0][0:-1]+id_components[1]+'D'
            if sum([each in checks[1]+checks[-1] for each in [id_components_check,id_components_check2,id_components[-1]]])==2:
                if items_f0[-1]!='-' and items_f0[-1]!='':
                    errors=items_f0[-1][1:-1].rsplit(',')
                    errors_number=[float(errors[0]),float(errors[1])]
                    items[0]=id_components[0][:-1]
                    if 'setdx' in items_f0[0]:
                        items[4]=items[4]+' ('+"{:.0E}".format(errors_number[0]*5.038)+', '+"{:.0E}".format(errors_number[1]*5.038)+')'
                        N_dxdydz+=1
                    elif 'setdy' in items_f0[0]:
                        items[5]=items[5]+' ('+"{:.0E}".format(errors_number[0]*5.434)+', '+"{:.0E}".format(errors_number[1]*5.434)+')'
                        N_dxdydz+=1
                    elif 'setdz' in items_f0[0]:
                        items[6]=items[6]+' ('+"{:.0E}".format(errors_number[0]*7.3707)+', '+"{:.0E}".format(errors_number[1]*7.3707)+')'
                        N_dxdydz+=1
    f2.write('\t'.join(items))
f0.close()
f1.close()
f2.close()

            
            
            
            
    