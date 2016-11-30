import os

#=======================================================================================================================
'''
WRITING RESULTS
'''
def save_results( data ):
    path = os.getcwd()
    with open( os.path.join(path, 'results.txt'), 'rw+') as f:
        data2 = f.read()
        data2 = '\n' + data#results2str(data)
        f.write(data2)
