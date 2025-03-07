#!/usr/bin/env python

"""
Introduction:
    MWCGS: Multiscale Weighted Colored Subgraphs

Author:
    Masud Rana (masud.rana@uky.edu)

Date last modified:
    Apr 7, 2023

"""

import sys
import os
import pandas as pd
from itertools import product
import ntpath
import argparse
import time

from agl_eat_score import *


class AlgebraicGraphLearningFeatures:

    df_kernels = pd.read_csv('../utils/kernels.csv')

    def __init__(self, args):
        """
        Parameters
        ----------
        kernel_index: int
            row index in kernels/kernels.csv
        cutoff: float
            distance cutoff to define binding site
        path_to_csv: str
            full path to csv data consisting of PDBID and Binding Affinity
        method: str
        	method to compute GGL features: either 'SYBYL' or 'ECIF'

        Output
        -------
        feature csv file

        """
        self.kernel_index = args.kernel_index
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv

        self.matrix_type = args.matrix_type

        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder


    def get_agl_features(self, parameters):

        df_pdbids = pd.read_csv(self.path_to_csv)
        pdbids = df_pdbids['PDBID'].tolist()
        pks = df_pdbids['pK'].tolist()

        Kernel = KernelFunction(kernel_type=parameters['type'],
                                kappa=parameters['power'], tau=parameters['tau'])


        AGL = AGL_EAT(Kernel=Kernel, cutoff=parameters['cutoff'], matrix_type=self.matrix_type)

        for index, _pdbid in enumerate(pdbids):
            lig_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_ligand.mol2'

            pro_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_protein.pdb'

            agl_score = AGL.get_agl_score(pro_file, lig_file)

            atom_pairs = agl_score['ATOM_PAIR'].tolist()
            features = agl_score.columns[1:].tolist()

            pairwise_features = [i[0]+'_'+i[1]
                                 for i in product(atom_pairs, features)]
            feature_values = agl_score.drop(
                ['ATOM_PAIR'], axis=1).values.flatten()
            if index == 0:
                df_features = pd.DataFrame(columns=[pairwise_features])
            df_features.loc[index] = feature_values

        df_features.insert(0, 'PDBID', pdbids)
        df_features.insert(1, 'pK', pks)

        return df_features

    def main(self):

        parameters = {
            'type': self.df_kernels.loc[self.kernel_index, 'type'],
            'power': self.df_kernels.loc[self.kernel_index, 'power'],
            'tau': self.df_kernels.loc[self.kernel_index, 'tau'],
            'cutoff': self.cutoff
        }

        df_features = self.get_agl_features(parameters)

        csv_file_name_only = ntpath.basename(self.path_to_csv).split('.')[0]

        output_file_name = f'{csv_file_name_only}_agl_eat_{self.matrix_type}_matrix_ker{self.kernel_index}_cutoff{self.cutoff}.csv'

        df_features.to_csv(f'{self.feature_folder}/{output_file_name}', index=False, float_format='%.5f')


def get_args(args):

    parser = argparse.ArgumentParser(description="Get AGL EAT Features")

    parser.add_argument('-k', '--kernel-index', help='Kernel Index (see kernels/kernels.csv)',
                        type=int)
    parser.add_argument('-c', '--cutoff', help='distance cutoff to define binding site',
                        type=float, default=12.0)
    parser.add_argument('-f', '--path_to_csv',
                        help='path to CSV file containing PDBIDs and pK values')
    parser.add_argument('-m', '--matrix_type', type=str,
    					help="type of graph matrix: either 'Laplacian' or 'Adjacency'",
    					default='Laplacian')
    parser.add_argument('-dd', '--data_folder', type=str,
    					help='path to data folder directory')
    parser.add_argument('-fd', '--feature_folder', type=str,
    					help='path to the directory where features will be saved')

    args = parser.parse_args()

    return args


def cli_main():
    args = get_args(sys.argv[1:])

    print(args)

    AGL_Features = AlgebraicGraphLearningFeatures(args)

    AGL_Features.main()


if __name__ == "__main__":

    t0 = time.time()

    cli_main()

    print('Done!')
    print('Elapsed time: ', time.time()-t0)
