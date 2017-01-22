import os

#=======================================================================================================================
'''
AUXILIARY FUNCTION FOR WRITING RESULTS
'''
def results2str( data ):
    for i in range(0,len(data)):
        if i == 0:
            s = str(data[i])
        else:
            s = s + '\t' + str(data[i])

    return s

'''
WRITING RESULTS
'''
def save_results( data ):
    path = os.getcwd()
    with open( os.path.join(path, 'results.txt'), 'rw+') as f:
        data2 = f.read()
        data2 = '\n' + data#results2str(data)
        f.write(data2)
