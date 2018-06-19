import os,inspect,copy,sys
SuPerRod_path=['/','\\'][int(os.name=='nt')].join(os.getcwd().rsplit(['/','\\'][int(os.name=='nt')])[0:os.getcwd().rsplit(['/','\\'][int(os.name=='nt')]).index('SuPerRod')+1])
print('SuPerRod_path @ '+ SuPerRod_path)
sys.path.append(SuPerRod_path)
sys.path.append(['/','\\'][int(os.name=='nt')].join([SuPerRod_path,'models']))
sys.path.append(['/','\\'][int(os.name=='nt')].join([SuPerRod_path,'lib']))
sys.path.append(['/','\\'][int(os.name=='nt')].join([SuPerRod_path,'plugins','data_loaders']))
import parameters,diffev,model
import filehandling as io
import numpy as np

##################################################################Tune the file here only################################################################
gx_file_name='P:\\temp_model\\CTR_Th_mica_LiCl_100mM_superrod_run1_Jun04.gx'                                                                                                                             #                                                                                                                                        #
                                                                                                                                                        #
path_of_data='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\CTR_RAXR_Th_LiCl_3mM_191217a_GenX_formate.dat'                                                                         #
data_loader='uaf_CTR_raxs_2_stand_alone'##can be easily customized from the data_loader plugin                                                          #
                                                                                                                                                        #
RAXR_EL='\'Th\''                                                                                                                                        #
wal=0.82657#wave length of x-ray                                                                                                                        #
E0=16300#adsorption energy of resonant element                                                                                                          #
F1F2_FILE='\'axd_ThLi_mica_Jun02_2018.f1f2\''#name of f1f2 file in the batch_file folder                                                                               #
                                                                                                                                                        #
c_lattice=20.1328                                                                                                                                     #
NUMBER_RAXS_SPECTRA=12                                                                                                                                  #
NUMBER_GAUSSIAN_PEAK=4 ##<11                                                                                                                            #
NUMBER_GAUSSIAN_PEAK_FREEZE=4##<11                                                                                                                      #
                                                                                                                                                        #
RAXR_FIT_MODE='\'MI\''                                                                                                                                  #
FREEZE='True'#True means the resonant element makes no contribution to the total CTR structure factor.                                                  #
                                                                                                                                                        #
#range setup for raxr datasets#                                                                                                                         #
a_range=[0,20]                                                                                                                                          #
b_range=[-5,5]                                                                                                                                          #
c_range=[0,10]                                                                                                                                          #
A_range=[0,50]                                                                                                                                           #                                                                                                                                  #                                                                                                                                       #
#########################################################################################################################################################

def local_func():
    return None

def script_path_locator(func=local_func):
    sub_folders=['standard_scripts','fit_structure_forward','differential evolution','muscovite001','genx_script_standard_RAXS_CTR_Muscovite.py']
    head_path=os.path.dirname(os.path.abspath(inspect.getsourcefile(func)))
    for folder in sub_folders:
        head_path=os.path.join(head_path,folder)
    return head_path

def dump_path_locator(func=local_func):
    sub_folders=['dump_files']
    head_path=os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(func))))
    for folder in sub_folders:
        head_path=os.path.join(head_path,folder)
    return head_path

def find_nth_overlapping(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+1)
        n -= 1
    return start


def make_script_file(path_locator=script_path_locator):
    script_path=path_locator()
    script_file=open(script_path)
    script=script_file.read()
    script_save=open(script_path.replace('.py','_cus.py'),'w')

    def _replace_script_section(str_text,key_text,replace_value,replace_loc=1):
        #loc=str_text.find(key_text,replace_loc-1)
        loc=find_nth_overlapping(str_text,key_text,replace_loc)
        comment_pos=None
        equal_sign_pos=None
        line_end_pos=None
        line_start_pos=loc
        while str_text[loc]!='\n':
            if str_text[loc]=='#':
                comment_pos=loc
            elif str_text[loc]=='=':
                equal_sign_pos=loc
            else:
                pass
            loc+=1
        line_end_pos=loc
        if comment_pos==None:
            comment_pos=line_end_pos
        text_to_be_feed=key_text+'='+str(replace_value)+str_text[comment_pos:line_end_pos]
        str_text=str_text.replace(str_text[line_start_pos:line_end_pos],text_to_be_feed,1)
        return str_text

    script=_replace_script_section(script,'RAXR_EL',RAXR_EL)
    script=_replace_script_section(script,'wal',wal)
    script=_replace_script_section(script,'E0',E0)
    script=_replace_script_section(script,'F1F2_FILE',F1F2_FILE)
    script=_replace_script_section(script,'c_lattice',c_lattice)
    script=_replace_script_section(script,'NUMBER_RAXS_SPECTRA',NUMBER_RAXS_SPECTRA)
    script=_replace_script_section(script,'NUMBER_GAUSSIAN_PEAK',NUMBER_GAUSSIAN_PEAK)
    script=_replace_script_section(script,'NUMBER_GAUSSIAN_PEAK_FREEZE',NUMBER_GAUSSIAN_PEAK_FREEZE)
    script=_replace_script_section(script,'RAXR_FIT_MODE',RAXR_FIT_MODE)
    script=_replace_script_section(script,'FREEZE',FREEZE)
    script=_replace_script_section(script,'a_range_raxr',a_range)
    script=_replace_script_section(script,'b_range_raxr',b_range)
    script=_replace_script_section(script,'c_range_raxr',c_range)
    script=_replace_script_section(script,'A_range_raxr',A_range)
    script_save.write(script)
    script_save.close()
    script_file.close()
    print('Script is saved!')
    return script

def make_par_file():
    mod = model.Model()
    config = io.Config()
    opt = diffev.DiffEv()
    mod.script=make_script_file()
    #mod.simulate()
    try:
        mod.simulate()
    except:
       pass

def make_data_file():
    exec(' '.join(['import', data_loader,'as data_processer']))
    data_handler=data_processer.Plugin()
    data_handler.LoadData(path_of_data)
    data_items_ctr=[]
    data_items_raxr=[]
    for each in data_handler.data.items[1:]:
        if each.x[0]>50:
            data_items_raxr.append(each)
        else:
            data_items_ctr.append(each)
    data_handler.data.items=data_items_ctr+data_items_raxr
    print 'Data file is created!'
    return data_handler.data,len(data_items_ctr)

def make_gx_file():
    mod = model.Model()
    config = io.Config()
    opt = diffev.DiffEv()
    par=parameters.Parameters()
    make_par_file()
    par.set_ascii_input_new(os.path.join(dump_path_locator(),'par_table.tab'))
    io.load_gx('temp_DONOT_delete.gx',mod,opt,config)
    mod.script=make_script_file()
    mod.parameters=par
    mod.data,num_ctr=make_data_file()
    for i in range(len(mod.data.items)-num_ctr):
        mod.data.items[i+num_ctr].use=False
    io.save_gx(gx_file_name,mod,opt,config)
    print('gx file is saved in the current workdir!\n')
    print('##########Basic model info##########')
    print('Resonant element: '+RAXR_EL)
    print('Number of Gaussian peaks in CTR: '+str(NUMBER_GAUSSIAN_PEAK))
    print('Number of Gaussian peaks in RAXR: '+str(NUMBER_GAUSSIAN_PEAK_FREEZE))
    print('Number of RAXR datasets: '+str(NUMBER_RAXS_SPECTRA))
    print('RAXR data fit mode: '+{'\'MI\''[1:3]:'Model independent fit','\'MD\''[1:3]:'Model dependent fit'}[RAXR_FIT_MODE[1:3]])


if __name__=='__main__':
    make_gx_file()
