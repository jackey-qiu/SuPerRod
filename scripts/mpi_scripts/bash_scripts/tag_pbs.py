import sys
#to use this script: python tag_pbs.py _May31.gx
tag = sys.argv[1]

file_names=['mpi_run_model_dependent.pbs','mpi_run_model_independent.pbs']

def find_nth_overlapping(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+1)
        n -= 1
    return start

def replace_script_section(str_text,key_text,replace_value,replace_loc=1):
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

for file in file_names:
    current_file=open(file).read()
    current_file=replace_script_section(current_file,'file','\''+tag+'\'')
    f=open(file,'w')
    f.write(current_file)
    f.close()

print('Tagging is finished as required!')
