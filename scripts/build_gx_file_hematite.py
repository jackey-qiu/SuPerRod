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
gx_file_name='P:\\temp_model\\Pb_anneal_rcut_run1_Jul19.gx'                                                                                                                             #                                                                                                                                         #
                                                                                                                                                        #
path_of_data='P:\\apps\\SuPerRod\\dump_files\\temp_full_dataset_pb_anneal.dat'                                                               #
data_loader='uaf_CTR_raxs_2_stand_alone'##can be easily customized from the data_loader plugin                                                          #
                                                                                                                                                        #
domain_setup_HLT=[3]#2 for short HLT, 3 for long HLT                                                                                                    #
domain_setup_FLT=[0,1]#0 for short FLT, 1 for long FLT                                                                                              #
water_layer_number=[4,4,4]                                                                                                                          #
atom_group_depth_one_domain=10                                                                                                                          #
atom_group_scheme_inter_domains=[[2,1]]#starting from 0 (0 for Domain1)                                                                           #
atom_group_depth_inter_domains=[[0,10]]                                                                                                          #
                                                                                                                                                        #
sorbate='Pb'                                                                                                                                            #
LOCAL_STRUCTURE_MATCH_LIB='{\'trigonal_pyramid\':[\'Pb\'],\'octahedral\':[\'Sb\',\'Fe\',\'Cd\',\'Cu\',\'Zn\'],\'tetrahedral\':[\'As\',\'P\',\'Cr\']}'   #
binding_mode=[['clean_H'],['TD_F'],['TD_F']]                                                                                  #
O_NUMBER=[[0,0],[0,0],[0,0]]                                                                                                                #
                                                                                                                                                        #
num_raxr_datasets=0                                                                                                                                    #
raxr_fit_mode='MI'                                                                                                                                      #
absorption_edge=11873                                                                                                                                   #
x_ray_wal=0.8625                                                                                                                                       #
f1f2_file_name='As_K_edge_March28_2018.f1f2'                                                                                                                         #
                                                                                                                                                        #
use_bv=1                                                                                                                                                #
bv_waiver=[]                                                                                                                                            #
#########################################################################################################################################################

def local_func():
    return None

def script_path_locator(func=local_func):
    sub_folders=['standard_scripts','fit_structure_forward','differential evolution','hematite_rcut','genx_script_standard_RAXS_CTR_hematite_rcut.py']
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

def binding_mode_translation(binding_mode,el=sorbate):
    mapping_lib={'CS_H':0,'CS2_H':1,'ES1_H':2,'ES2_H':3,'TD_H':4,'TDFS_H':5,'OS_H':6,'clean_H':7,\
                 'CS_F':15,'CS2_F':18,'ES1_F':16,'ES2_F':17,'TD_F':19,'TDFS_F':20,'OS_F':21,'clean_F':22}
    binding_mode_num=copy.deepcopy(binding_mode)
    binding_el=copy.deepcopy(binding_mode)
    for i in range(len(binding_mode)):
        for j in range(len(binding_mode[i])):
            binding_mode_num[i][j]=mapping_lib[binding_mode[i][j]]
            binding_el[i][j]=el
    return binding_mode_num,binding_el

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

    script=_replace_script_section(script,'USE_BV',use_bv)
    script=_replace_script_section(script,'half_layer',domain_setup_HLT)
    script=_replace_script_section(script,'full_layer',domain_setup_FLT)
    script=_replace_script_section(script,'GROUPING_SCHEMES',atom_group_scheme_inter_domains)
    script=_replace_script_section(script,'GROUPING_DEPTH',atom_group_depth_inter_domains)
    script=_replace_script_section(script,'pickup_index',binding_mode_translation(binding_mode,sorbate)[0],2)
    script=_replace_script_section(script,'SORBATE',binding_mode_translation(binding_mode,sorbate)[1])
    script=_replace_script_section(script,'LOCAL_STRUCTURE_MATCH_LIB',LOCAL_STRUCTURE_MATCH_LIB)
    script=_replace_script_section(script,'O_NUMBER',O_NUMBER,5)
    script=_replace_script_section(script,'WATER_LAYER_NUM',water_layer_number)
    script=_replace_script_section(script,'RAXR_EL','\''+sorbate+'\'')
    script=_replace_script_section(script,'RAXR_FIT_MODE','\''+raxr_fit_mode+'\'')
    script=_replace_script_section(script,'NUMBER_SPECTRA',num_raxr_datasets)
    script=_replace_script_section(script,'E0',absorption_edge)
    script=_replace_script_section(script,'wal',x_ray_wal)
    script=_replace_script_section(script,'F1F2_FILE','\''+f1f2_file_name+'\'')
    commands_other=[]
    for i in range(len(binding_mode)):
        if not binding_mode[i][0].startswith('clean'):
            commands_other.append('_'.join(['gp','HO','set2','D'+str(i+1)+'.setu'])+'('+'_'.join(['gp','HO','set1','D'+str(i+1)+'.getu()'])+')')
            commands_other.append('_'.join(['gp','sorbates','set2','D'+str(i+1)+'.setoc'])+'('+'_'.join(['gp','sorbates','set1','D'+str(i+1)+'.getoc()'])+')')
    script=_replace_script_section(script,'commands_other',commands_other)
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
    par.set_ascii_input_new(os.path.join(dump_path_locator(),'table.tab'))
    io.load_gx('temp_DONOT_delete.gx',mod,opt,config)
    mod.script=make_script_file()
    mod.parameters=par
    mod.data,num_ctr=make_data_file()
    for i in range(len(mod.data.items)-num_ctr):
        mod.data.items[i+num_ctr].use=False
    io.save_gx(gx_file_name,mod,opt,config)
    print('gx file is saved in the current workdir!\n')
    print('##########Basic model info##########')
    print('Domain number: '+str(len(domain_setup_HLT))+' half layer domains and '+str(len(domain_setup_FLT))+' full layer domains')
    print('sorbate: '+sorbate)
    local_structure_db=eval(LOCAL_STRUCTURE_MATCH_LIB)
    for key in local_structure_db.keys():
        if sorbate in local_structure_db[key]:
            print('local_structure: '+key)
            break
        else:
            pass
    binding_mode_db=[]
    for i in range(len(binding_mode)):
        if binding_mode[i][0].startswith('clean'):
            binding_mode_db.append('Clean_surface')
        elif binding_mode[i][0].startswith('CS'):
            binding_mode_db.append('Corner-sharing')
        elif binding_mode[i][0].startswith('ES'):
            binding_mode_db.append('Edge-sharing')
        elif binding_mode[i][0].startswith('TD'):
            binding_mode_db.append('Tridentate-binding')
        elif binding_mode[i][0].startswith('OS'):
            binding_mode_db.append('Outersphere binding')
    print('Binding_mode: '+','.join(binding_mode_db))


if __name__=='__main__':
    make_gx_file()
