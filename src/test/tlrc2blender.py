'''
Created on Dec 1, 2014

@author: noampeled
'''

from src.commons.utils import utils
from src.commons.utils import plots

TLRC_FILE_NAME = '/Users/noampeled/Documents/blender/dor_cubes_trlc.mat'
CUBES_FILE_NAME = '/Users/noampeled/Copy/Data/MEG/data/svmFiles/dataForDecoding.mat_centipedeSpacialSWFreqs_StayOrLeave6_10_features__subdor_cubes.mat'
BLENDER_FILE_NAME = '/Users/noampeled/Documents/blender/cubes.csv'
SAM_STEP_SIZE = 1.5  # cm


def convert():
    r = SAM_STEP_SIZE
    tlrc = utils.loadMatlab(TLRC_FILE_NAME)['cubes']
    cubes = utils.loadMatlab(CUBES_FILE_NAME)['cubes']
    cubes[:, :3] = tlrc[:, :3]
    with open(BLENDER_FILE_NAME, 'wb') as bFile:
        file_writer = utils.csv.writer(bFile, delimiter=',')
        colors = plots.arrToColors(cubes[:, 3])
        r /= 2.
        cubes[:, :3] *= 10.0
        for index, (cube, c) in enumerate(zip(cubes, colors)):
            cube[:3] /= float(10)
            file_writer.writerow([index, 'q', cube[0], cube[1], cube[2],
                r, c[0], c[1], c[2], '', '', '', '', 0.5])

if __name__ == '__main__':
    convert()
    print('finish!')
