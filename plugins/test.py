import data_loaders

def func():
    __import__('data_loaders.default',globals(),locals(),[])
    print('import data_loaders')
    pass
func()