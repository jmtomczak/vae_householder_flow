from VAE_input_mnist import openMnist
from VAE_input_histopathology import openHistopathology

def openDataset(datasetName, small=False):
    if datasetName == 'mnist':
        return openMnist(small=small)
    elif datasetName == 'histopathology':
        return openHistopathology()
    else:
        raise Exception('dataset not found')
