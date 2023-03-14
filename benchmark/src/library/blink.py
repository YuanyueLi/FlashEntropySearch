#!/usr/bin/env python

import sys
import os
import argparse
import glob
from timeit import default_timer as timer
import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import pandas as pd
# from pyteomics import mgf
import pymzml
import networkx as nx

import matplotlib.pyplot as plt
###########################
# Mass Spectra Transforms
###########################
def filter_spectra(mzis,pmzs,pmz_delta=0.05):
    """
    remove zero intensities to keep the sparse arrays from breaking
    """
    cleaned_mzis = []
    for i,mzi in enumerate(mzis):
        idx = np.argwhere(mzi[1]>0).flatten()
        cleaned_mzis.append(mzi[:,idx])
    return cleaned_mzis

def remove_duplicate_ions(mzis, min_diff=0.002):
    """
    remove peaks from a list of 2xM mass spectrum vectors (mzis) that are within min_diff
    by averaging m/zs and summing intensities to remove duplicates

    options:
        min_diff, float
            minimum difference possible given,
            a good rule of thumb is ions have to be greater than 2 bins apart

    returns:
        mzis, list of 2xM mass spectrum vectors
    """
    bad_ones = [i for i,s in enumerate(mzis) if min(np.diff(s[0],prepend=0))<=min_diff]
    for bad_idx in sorted(bad_ones, reverse=True):
        idx = np.argwhere(np.diff(mzis[bad_idx][0])<min_diff).flatten()
        idx = sorted(idx,reverse=True)
        for sub_idx in idx:
            dup_mz = mzis[bad_idx][0][sub_idx:sub_idx+2]
            dup_intensities = mzis[bad_idx][1][sub_idx:sub_idx+2]
            new_mz = np.mean(dup_mz)
            new_intensity = np.sum(dup_intensities)
            mzis[bad_idx][0][sub_idx:sub_idx+2] = new_mz
            mzis[bad_idx][1][sub_idx:sub_idx+2] = new_intensity
        mz = np.delete(mzis[bad_idx][0],idx)
        intensity = np.delete(mzis[bad_idx][1],idx)
        mzis[bad_idx] = np.asarray([mz,intensity])

    return mzis

def discretize_spectra(mzis, pmzs, bin_width=0.001, intensity_power=0.5, trim_empty=False, remove_duplicates=False, metadata=None, calc_network_score=False):
    """
    converts a list of 2xM mass spectrum vectors (mzis) and pmzs into a dict-based sparse matrix of [mz/nl][i/c] components

    options:
        bin_width, float
            width of bin to use in mz
        intensity_power, float
            power to raise intensity to before normalizing
        trim_empty, bool
            remove empty spectra from list
        remove_duplicates, bool
            average mz and intensity over peaks within 2 times bin_width
        calc_network_score, bool
            calculate and store neutral loss matrices

    returns:
        {'i',
         'c',
         'spec_ids',
         'mz',
         'pmz',
         'shift',
         'bin_width',
         'intensity_power',
          'metadata'}
          
          optional:
          {'nl',
           'blanks'}
    """
    mzis = filter_spectra(mzis,pmzs) 

    if trim_empty:
        kept, mzis = np.array([[idx,mzi] for idx,mzi in enumerate(mzis) if mzi.size>0], dtype=object).T
    if remove_duplicates:
        mzis = remove_duplicate_ions(mzis,min_diff=bin_width*2)
    
    spec_ids = np.concatenate([[i]*mzi.shape[1] for i,mzi in enumerate(mzis)]).astype(int)

    inorm = np.array([1./np.linalg.norm(mzi[1]**intensity_power) for mzi in mzis])
    cnorm = np.ones(len(mzis))
    num_ions = [mzi.shape[1] for mzi in mzis]

    if metadata is None:
        metadata = [{} for ele in range(len(mzis))]
        
    for i,m in enumerate(mzis):
        metadata[i]['num_ions'] = num_ions[i]

    mzis = np.concatenate(mzis, axis=1)
    
    mzis[1] = mzis[1]**intensity_power 
    normalized_intensities = inorm[spec_ids]*mzis[1]
    normalized_counts = cnorm[spec_ids].astype(int)
    
    mz_bins = np.rint(mzis[0]/bin_width).astype(int)
    
    if calc_network_score == True:
        nl_bins = (np.rint(np.asarray(pmzs)[spec_ids]/bin_width) - mz_bins).astype(int)
        shift = -nl_bins.min().astype(int) #to remove negative numbers and zeros from the bins 
        nl_bins = nl_bins+shift
        
    else:
        shift = 0

    mz_bins = mz_bins+shift

    S = {'i': normalized_intensities,
        'c': normalized_counts,
        'spec_ids': spec_ids,
        'mz': mz_bins,
        'pmz': pmzs,
        'shift':shift,
        'bin_width': bin_width,
        'intensity_power': intensity_power,
        'metadata':metadata}
    
    if calc_network_score == True:
        S['nl'] = nl_bins

    if trim_empty:
        S['blanks'] = np.setdiff1d(np.arange(spec_ids[-1]+1),kept)

    return S

def maximum_entropy_normalization(y):
    #MaximumEntropyNormalization   Normalize Data using principle of maximum entropy.
    # n=1/(1+exp(-(y-mean(y))/std(y)));
    m=y.mean()
    s=y.std()
    n=1/(1+np.exp(-1*(y-m)/s))
    return n

##########
# Kernel
##########
def network_kernel(S, tolerance=0.01, mass_diffs=[0], react_steps=1, calc_network_score=False):
    """
    apply network kernel to all mzs/nls in S that are within
    tolerance of any combination of mass_diffs within react_steps

    options:
        tolerance, float
            tolerance in mz from mass_diffs for networking ions
        mass_diffs, listlike of floats
            mass differences to consider networking ions
        react_steps, int
            expand mass_diffs by the +/- combination of all mass_diffs within
            specified number of reaction steps

    returns:
        S + {'intensity_net',
             'count_net',
             'spec_ids_net',
             'mz_net'}
    """
    bin_num = int(2*(tolerance/S['bin_width'])-1)

    mass_diffs = np.sort(np.abs(mass_diffs))

    mass_diffs = [-m for m in mass_diffs[::-1]]+[m for m in mass_diffs]
    if mass_diffs[len(mass_diffs)//2] == 0:
        mass_diffs.pop(len(mass_diffs)//2)

    mass_diffs = np.rint(np.array(mass_diffs)/S['bin_width']).astype(int)

    # Recursively "react" mass_diffs within a specified number of reation steps
    def react(mass_diffs, react_steps):
        if react_steps == 1:
            return mass_diffs
        else:
            return np.add.outer(mass_diffs, react(mass_diffs, react_steps-1))

    # Expand reacted mass_diffs to have a tolerance
    mass_diffs = np.unique(np.sort(react(mass_diffs, react_steps).flatten()))
    mass_diffs = np.add.outer(mass_diffs, np.arange(-bin_num//2+1, bin_num//2+1)).flatten()
    
    S['i_net'] = np.add.outer(S['i'], np.zeros_like(mass_diffs)).flatten()
    S['c_net'] = np.add.outer(S['c'], np.zeros_like(mass_diffs)).flatten()
    S['spec_ids_net'] = np.add.outer(S['spec_ids'], np.zeros_like(mass_diffs)).flatten()
    S['mz_net'] = np.add.outer(S['mz'], mass_diffs).flatten()
    S['shift_net'] = S['shift']
    
    if calc_network_score == True:

        S['nl_net'] = np.add.outer(S['nl'], mass_diffs).flatten()
        S['shift_net'] = S['shift']-S['nl_net'].min()
        S['mz_net'] -= S['nl_net'].min()
        S['nl_net'] -= S['nl_net'].min()
        
    return S

#####################
# Biochemical Masses
#####################

biochem_masses = [0.,      # Self
                  12.,     # C
                  1.00783, # H
                  2.01566, # H2
                  15.99491,# O
                  0.02381, # NH2 - O
                  78.95851,# PO3
                  31.97207]# S

############################
# Comparing Sparse Spectra
############################

def construct_sparse_matrices(S, networked=False, calc_network_score=False):
    
    if networked:
        networked = '_net'
    else:
        networked = ''

    E = {'mzi': sp.coo_matrix((S['i'+networked], (S['mz'+networked], S['spec_ids'+networked])), dtype=float, copy=False),
        'mzc': sp.coo_matrix((S['c'+networked], (S['mz'+networked], S['spec_ids'+networked])), dtype=int,   copy=False)}
    
    if calc_network_score == True:
        E['nli'] = sp.coo_matrix((S['i'+networked], (S['nl'+networked], S['spec_ids'+networked])), dtype=float, copy=False)
        E['nlc'] = sp.coo_matrix((S['c'+networked], (S['nl'+networked], S['spec_ids'+networked])), dtype=int,   copy=False)
    
    return E

def score_sparse_spectra(S1, S2, tolerance=0.01, mass_diffs=[0], react_steps=1, calc_network_score=False):
    """
    score/match/compare two sparse mass spectra

    tolerance, float
        tolerance in mz from mass_diffs for networking ions
    mass_diffs, listlike of floats
        mass differences to consider networking ions
    react_steps, int
        expand mass_diffs by the +/- combination of all mass_diffs within
        specified number of reaction steps

    returns:
        S1 vs S2 scores, scipy.sparse.csr_matrix
    """
    # Expand complex valued sparse matrices into [mz/nl][i/c] matrices
    
    ordered = S1['c'].size < S2['c'].size

    network_kernel(S1 if ordered else S2, tolerance, mass_diffs, react_steps, calc_network_score=calc_network_score)
    
    E1 = construct_sparse_matrices(S1, networked=ordered, calc_network_score=calc_network_score)
    E2 = construct_sparse_matrices(S2, networked=not ordered, calc_network_score=calc_network_score)

    S1_shift = S1['shift_net'] if ordered else S1['shift']
    S2_shift = S2['shift'] if ordered else S2['shift_net']

    # Return score/matches matrices for mzs/nls
    S12 = {}
    for k in set(E1.keys()) & set(E2.keys()):
        v1, v2 = E1[k].T, E2[k]

        if S1_shift <= S2_shift:
            v1 = sp.hstack([sp.coo_matrix((v1.shape[0], S2_shift-S1_shift), dtype=v1.dtype), v1], format='csr', dtype=v1.dtype)
        if S2_shift < S1_shift:
            v2 = sp.vstack([sp.coo_matrix((S1_shift-S2_shift, v2.shape[1]), dtype=v2.dtype), v2], format='csc', dtype=v2.dtype)

        max_mz = max(v1.shape[1],v2.shape[0])
        v1.resize((v1.shape[0],max_mz))
        v2.resize((max_mz,v2.shape[1]))

        S12[k] = v1.tocsr().dot(v2.tocsc())
    S12['S1_metadata'] = S1['metadata']
    S12['S2_metadata'] = S2['metadata']

    return S12

def compute_network_score(S12):
    S12['network_score'] = S12['mzi'].maximum(S12['nli'])
    S12['network_matches'] = S12['mzc'].maximum(S12['nlc'])
    return S12


def filter_hits(S12,good_score=0.5,min_matches=5,good_matches=20,calc_network_score=False):
    """
    filter mzi and nli scores
    filter mzc and nlc counts

    returns filtered score and network score as coo matrices

    good_score = 0.5 #remove hits unless count >= good_matches
    min_matches = 5 # completely ignore any hits with less than this many matches
    good_matches = 20 #remove hits unless score >= good_score

    """
    
    if calc_network_score==True:
        S12 = compute_network_score(S12)
        #filter on blink network score
        idx = S12['network_score']>=good_score
        if min_matches is not None:
            idx = idx.multiply(S12['network_matches']>=min_matches)
        if good_matches is not None:
            idx = idx.maximum(S12['network_matches']>=good_matches)
        S12['network_score'] = S12['network_score'].multiply(idx).tocoo()
        S12['network_matches'] = S12['network_matches'].multiply(idx).tocoo()
        # S12['mzi'] = S12['mzi'].multiply(idx).tocoo()
        # S12['mzc'] = S12['mzc'].multiply(idx).tocoo()
        # S12['nli'] = S12['nli'].multiply(idx).tocoo()
        # S12['nli'] = S12['nli'].multiply(idx).tocoo()
    else:
        # filter on blink score
        idx = S12['mzi']>=good_score
        if min_matches is not None:
            idx = idx.multiply(S12['mzc']>=min_matches)
        if good_matches is not None:
            idx = idx.maximum(S12['mzc']>=good_matches)
        # S12['network_score'] = S12['network_score'].multiply(idx).tocoo()
        # S12['network_matches'] = S12['network_matches'].multiply(idx).tocoo()
        S12['mzi'] = S12['mzi'].multiply(idx).tocoo()
        S12['mzc'] = S12['mzc'].multiply(idx).tocoo()
        # S12['nli'] = S12['nli'].multiply(idx).tocoo()
        # S12['nli'] = S12['nli'].multiply(idx).tocoo()
        
    return S12

def get_topk_blink_matrix(D,k=5,score_col=4,query_col=1):
    """
    sort by score_col and query_col
    get top k for each query_col entry

    returns filtered blink matrix
    """

    # Do a quick hand calc to make sure this is working:
    # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.sort.html
    D = D[np.lexsort((D[:, query_col], -D[:, score_col])),:]
    # idx = np.argsort(D[:,score_col])[::-1] #sort in descending order by blink score
    # D = D[idx,:]
    # idx = np.argsort(D[:,query_col]) # sort in ascending order by query number
    # D = D[idx,:]



    #return indices of first instance of each query number
    u,u_idx = np.unique(D[:,query_col],return_index=True)

    #for each query number get k next best hits
    hits = []
    for i,idx in enumerate(u_idx[:-1]):
        if (u_idx[i+1]-idx)>=k: #check the >=
            hits.append(np.arange(k)+idx)
        else:
            hits.append(np.arange(u_idx[i+1]-idx)+idx)
    hits = np.concatenate(hits)
    D = D[hits,:]
    return D

def create_blink_matrix_format(S12,calc_network_score=False):
    """
    reshape score and matches matrices such that they can be associated with metadata
    """

    if calc_network_score==True:
        # idx_nl = np.ravel_multi_index((S12['nli']f,S12['nli'].col),S12['nli'].shape)
        idx = np.ravel_multi_index((S12['network_score'].row,S12['network_score'].col),S12['network_score'].shape)
        # idx = np.unique(np.concatenate([idx_mz,idx_nl]))
        r,c = np.unravel_index(idx,S12['network_score'].shape)
    else:
        idx = np.ravel_multi_index((S12['mzi'].row,S12['mzi'].col),S12['mzi'].shape)
        r,c = np.unravel_index(idx,S12['mzi'].shape)

    M = np.zeros((len(idx),5))#,dtype='>i4')
    M[:,0] = idx
    M[:,1] = r #query
    M[:,2] = c #reference
    if calc_network_score==True:
        idx = np.in1d(idx, idx).nonzero()
        M[idx,3] = S12['network_score'].data
        M[idx,4] = S12['network_matches'].data
    else:
        idx = np.in1d(idx, idx).nonzero()
        M[idx,3] = S12['mzi'].data
        M[idx,4] = S12['mzc'].data

    #remove self connections
    M = M[M[:,1]!=M[:,2]]

    return M

#######################
# Mass Spectra Loading
#######################

def read_mzml(filename):
    """
    Takes in the full path to an mzml file

    For files with MS2, returns spectra, precursor m/z, intensity,
    and retention time

    For files with MS^n, returns the above plus relationships to the
    spectrum collection and what the particular precursor was.

    returns a dataframe that can go into the downstream processes.
    """
    def make_spectra(tuple_spectrum):
        mzs = []
        intensities = []
        for m,i in tuple_spectrum:
            mzs.append(m)
            intensities.append(i)
        np_spectrum = np.asarray([mzs,intensities])
        return np_spectrum

    precision_dict = {}
    for i in range(100):
        precision_dict[i] = 1e-5

    run = pymzml.run.Reader(filename, MS_precisions=precision_dict)
    spectra = list(run)

    df = []
    for s in spectra:
        
        if s.ms_level is None:
            continue

        if s.ms_level>=2:
            for precursor_dict in s.selected_precursors:
                data = {'id':s.ID,
                        'ms_level':s.ms_level,
                        'rt':s.scan_time_in_minutes(),
                        'spectrum':s.peaks('centroided')}
                if precursor_dict['precursor id'] is not None:
                    for k in precursor_dict.keys():
                        data[k] = precursor_dict[k]
                        # print(k,precursor_dict[k])
                df.append(data)
    df = pd.DataFrame(df)
    df.dropna(subset=['precursor id'],inplace=True)

    df['spectrum'] = df['spectrum'].apply(make_spectra)
    df['id'] = df['id'].astype(int)
    df['precursor id'] = df['precursor id'].astype(int)

    if df['ms_level'].max()>2:
        G=nx.from_pandas_edgelist(df, source='precursor id', target='id')
        # get the collection of spectra
        sub_graph_indices=list(nx.connected_components(G))
        # expand to [<spec. collection>,<id>]
        sub_graph_indices = [(i, v) for i,d in enumerate(sub_graph_indices) for k, v in enumerate(d)]
        sub_graph_indices = pd.DataFrame(sub_graph_indices,columns=['spectrum_collection','id'])
        df = pd.merge(df,sub_graph_indices,left_on='id',right_on='id',how='left')
        prec_mz_df = df[df['ms_level']==2].copy()
        prec_mz_df.rename(columns={'mz':'root_precursor_mz',
                                   'i':'root_precursor_intensity',
                                   'rt':'root_precursor_rt'},inplace=True)
        df.drop(columns=['i'],inplace=True)
        df.rename(columns={'mz':'precursor_mz'},inplace=True)
        df = pd.merge(df,
                      prec_mz_df[['spectrum_collection',
                                  'root_precursor_mz',
                                  'root_precursor_intensity','root_precursor_rt']],
                      left_on='spectrum_collection',
                      right_on='spectrum_collection')
        df.rename(columns={},inplace=True)
        # df.sort_values('rt',inplace=True)
        # df.drop_duplicates('rt',inplace=True)
        df.reset_index(inplace=True,drop=True)
        # df.head(30)
    else:
        df.drop(columns=['precursor id'],inplace=True)
        df.rename(columns={'mz':'precursor_mz'},inplace=True)
        df.reset_index(inplace=True,drop=True)
    return df

def read_mgf(in_file):
    msms_df = []
    with mgf.MGF(in_file) as reader:
        for spectrum in reader:
            d = spectrum['params']
            d['spectrum'] = np.array([spectrum['m/z array'],
                                      spectrum['intensity array']])
            if 'precursor_mz' not in d:
                d['precursor_mz'] = d['pepmass'][0]
            else:
                d['precursor_mz'] = float(d['precursor_mz'])
            msms_df.append(d)
    msms_df = pd.DataFrame(msms_df)
    return msms_df

def write_sparse_msms_file(out_file, S):
    np.savez_compressed(out_file, **S)

def open_msms_file(in_file):
    if '.mgf' in in_file:
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        return read_mgf(in_file)
    if '.mzml' in in_file.lower():
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        return read_mzml(in_file)
    else:
        logging.error('Unsupported file type: {}'.format(os.path.splitext(in_file)[-1]))
        raise IOError

def open_sparse_msms_file(in_file):
    if '.npz' in in_file:
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        with np.load(in_file, mmap_mode='w+',allow_pickle=True) as S:
            return dict(S)
    else:
        logging.error('Unsupported file type: {}'.format(os.path.splitext(in_file)[-1]))
        raise IOError
        
#########################
# TOPOLOGY FILTERS. Most From GNPS Quickstart Github Repo
#########################

def filter_component_additive(G, max_component_size,edge_score='score'):
    if max_component_size == 0:
        return

    all_edges = list(G.edges(data=True))
    G.remove_edges_from(list(G.edges))

    all_edges = sorted(all_edges, key=lambda x: x[2][edge_score], reverse=True)
    counter = 0
    for i,edge in enumerate(all_edges):
        G.add_edges_from([edge])
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > max_component_size:
            G.remove_edge(edge[0], edge[1])
        counter +=1
        if counter==1000:
            print(i,len(list(G.edges(data=True))))
            counter=0
        
def filter_top_k(G, top_k,edge_score='score'):
    print("Filter Top_K", top_k)
    #Keeping only the top K scoring edges per node
    print("Starting Numer of Edges", len(G.edges()))

    node_cutoff_score = {}
    for node in G.nodes():
        node_edges = G.edges((node), data=True)
        node_edges = sorted(node_edges, key=lambda edge: edge[2][edge_score], reverse=True)

        edges_to_delete = node_edges[top_k:]
        edges_to_keep = node_edges[:top_k]

        if len(edges_to_keep) == 0:
            continue

        node_cutoff_score[node] = edges_to_keep[-1][2][edge_score]

        #print("DELETE", edges_to_delete)


        #for edge_to_remove in edges_to_delete:
        #    G.remove_edge(edge_to_remove[0], edge_to_remove[1])


    #print("After Top K", len(G.edges()))
    #Doing this for each pair, makes sure they are in each other's top K
    edge_to_remove = []
    for edge in G.edges(data=True):
        cosine_score = edge[2][edge_score]
        threshold1 = node_cutoff_score[edge[0]]
        threshold2 = node_cutoff_score[edge[1]]

        if cosine_score < threshold1 or cosine_score < threshold2:
            edge_to_remove.append(edge)

    for edge in edge_to_remove:
        G.remove_edge(edge[0], edge[1])

    print("After Top K Mutual", len(G.edges()))
    
#########################
# Convenient Task Runner for Most Common Use Cases
#########################
def get_blink_hits(query,ref,calc_network_score=False,
                  min_matches=5,good_matches=20,good_score=0.5,precursor_match=5, tolerance=0.01):
    if isinstance(query, str):
        query = open_msms_file(query)

    if 'spectrum' in query.columns:
        if isinstance(query, pd.DataFrame):
            query = discretize_spectra(query['spectrum'].tolist(),pmzs=query['precursor_mz'].tolist(),
                                         remove_duplicates=False,metadata=query.drop(columns=['spectrum']).to_dict(orient='records'), calc_network_score=calc_network_score)
        if isinstance(ref, pd.DataFrame):
            ref = discretize_spectra(ref['spectrum'].tolist(),pmzs=ref['precursor_mz'].tolist(),
                                         remove_duplicates=False,metadata=ref.drop(columns=['spectrum']).to_dict(orient='records'), calc_network_score=calc_network_score)
            
        S12 = score_sparse_spectra(query, ref, tolerance=tolerance, calc_network_score=calc_network_score)
        S12 = filter_hits(S12,min_matches=min_matches,good_matches=good_matches,good_score=good_score,calc_network_score=calc_network_score)
        D = create_blink_matrix_format(S12,calc_network_score=calc_network_score)
        df = pd.DataFrame(D,columns=['raveled_index','query','ref','score','matches'])

        df = pd.merge(df,pd.DataFrame(S12['S1_metadata']).add_suffix('_query'),left_on='query',right_index=True,how='left')
        df = pd.merge(df,pd.DataFrame(list(S12['S2_metadata'])).add_suffix('_ref'),left_on='ref',right_index=True,how='left')
        
        #  TO DO!!!!
        #
        # If precursor_match is not False then there are ways 100x faster than this to deal with the filter.
        # The most important thing is to do it right AFTER SCORE_SPARSE_SPECTRA
        # 
        # This works
        # def f(a,b,c):
        #     out = ne.evaluate('C * abs(A+B)<0.01',{'A':a[...,None],'B':b,'C':c})
        #     return out
        # a = [m['precursor_mz'] for m in query['metadata']]
        # a = np.asarray(a)
        # out = f(a,a,blink_score['mzi'].todense())


        # best to do a check that precursor mz is the actual name of attribute or there will be trouble
        if ('precursor_mz_query' in df.columns) & ('precursor_mz_ref' in df.columns):
            df['precursor_ppm_diff'] = 1e6 * abs(df['precursor_mz_query'] - df['precursor_mz_ref'])/df['precursor_mz_query']
            if precursor_match is not False:
                idx = df['precursor_ppm_diff']<precursor_match
                df = df[idx]
        else:
            print('you can not filter by precursor mz since it is not there')
            
        if ('num_ions_query' in df.columns) & ('num_ions_ref' in df.columns):
            df['jaccard_matches'] = df['matches'] / ((df['num_ions_query']+df['num_ions_ref'])-df['matches'])
            df['overlap_matches'] = df['matches'] / df[['num_ions_query','num_ions_ref']].min(axis=1)
        else:
            print('you can not calculate overlap matches since it is not there')
            
        df['score_rank'] = df.groupby("query")["score"].rank("dense", ascending=False)
        df['matches_rank'] = df.groupby("query")["matches"].rank("dense", ascending=False)
        if 'jaccard_matches' in df.columns:
            df['jaccard_matches_rank'] = df.groupby("query")["jaccard_matches"].rank("dense", ascending=False)

        return df
    else:
        return None

#########################
# Convenience Plotting Functions
#########################

def make_mirror_plot(r,q,figsize=(12,6),mz_tol=0.01,color='black',match_color='red',fontsize=20,grid=True,ax=None,fig=None,vshade=None):
    """
    Make a mirror plot
    You should normalize the spectra before passing to this.
    q = query spectrum np.array([mzs,intensities])
    r = reference spectrum np.array([mzs,intensities])
    returns figure and axes handles
    """
    # return_fig = False
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
        # return_fig = True
        
    dd = abs(np.subtract(r[0][None,:], q[0][:, None],))<mz_tol

    ax.vlines(q[0],q[1]*0,q[1],color,alpha=0.6,linewidth=2)
    idx = dd.any(axis=1)
    ax.vlines(q[0][idx],q[1][idx]*0,q[1][idx],match_color,linewidth=2)

    ax.vlines(r[0],r[1]*0,-1*r[1],color,alpha=0.6,linewidth=2)
    idx = dd.any(axis=0)
    ax.vlines(r[0][idx],r[1][idx]*0,-1*r[1][idx],match_color,linewidth=2)
    ax.set_ylabel('Normalized intensity (au)',fontsize=fontsize)
    ax.set_xlabel('m/z',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-2))
    if vshade is not None:
        ax.axvline(vshade[0],linewidth=vshade[1],alpha=vshade[2],color=vshade[3])
    if grid==True:
        ax.grid()
    # if return_fig==True
    # return fig,ax

#########################
# Command Line Interface
#########################

'''
https://stackoverflow.com/questions/4194948/python-argparse-is-there-a-way-to-specify-a-range-in-nargs
unutbu
'''
def required_length(nmin,nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values)<=nmax:
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

def arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='BLINK discretizes mass spectra (given .mgf inputs), and scores discretized spectra (given .npz inputs)')

    parser.add_argument('files',nargs='+', action=required_length(1,2), metavar='F', help='files to process')

    #Discretize options
    discretize_options = parser.add_argument_group()
    discretize_options.add_argument('--trim', action='store_true', default=False, required=False,
                                    help='remove empty spectra when discretizing')
    discretize_options.add_argument('--dedup', action='store_true', default=False, required=False,
                                    help='deduplicate fragment ions within 2 times bin_width')
    discretize_options.add_argument('-b','--bin_width', type=float, metavar='B', default=0.001, required=False,
                                 help='width of bins in mz')
    discretize_options.add_argument('-i','--intensity_power', type=float, metavar='I', default=0.5, required=False,
                                 help='power to raise intensites to in when scoring')

    #Compute options
    compute_options = parser.add_argument_group()
    compute_options.add_argument('-t','--tolerance', type=float, metavar='T', default=0.01, required=False,
                                 help='maximum tolerance in mz for fragment ions to match')
    compute_options.add_argument('-d','--mass_diffs', type=float, metavar='D', nargs='*', default=[0], required=False,
                              help='mass diffs to network')
    compute_options.add_argument('-r','--react_steps', type=int, metavar='R', default=1, required=False,
                              help='recursively combine mass_diffs within number of reaction steps')
    compute_options.add_argument('-s','--min_score', type=float, default=0.4, metavar='S', required=False,
                                 help='minimum score to include in output')
    compute_options.add_argument('-m','--min_matches', type=int, default=3, metavar='M', required=False,
                                 help='minimum matches to include in output')

    #Output file options
    output_options = parser.add_argument_group()
    output_options.add_argument('--fast_format', action='store_true', default=False, required=False,
                                help='use fast .npz format to store scores instead of .tab')
    output_options.add_argument('-f', '--force', action='store_true', required=False,
                                help='force file(s) to be remade if they exist')
    output_options.add_argument('-o','--out_dir', type=str, metavar='O', required=False,
                                help='change output location for output file(s)')

    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(os.getcwd(),'blink.log'), level=logging.INFO)

    common_ext = {os.path.splitext(in_file)[1]
                  for f in args.files
                  for in_file in glob.glob(f)}
    if len(common_ext) == 1:
        common_ext = list(common_ext)[0]

    if common_ext == '.mgf':
        logging.info('Discretize Start')

        files = [os.path.splitext(os.path.splitext(
                 os.path.basename(in_file))[0])[0]
                 for input in args.files
                 for in_file in glob.glob(input)]

        prefix = os.path.commonprefix(files)

        if len(files) > 1:
            out_name = '-'.join([files[0],files[-1]])
        else:
            out_name = files[0]

        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = os.path.dirname(os.path.abspath(glob.glob(args.files[0])[0]))

        logging.info('Output to {}'.format(out_dir))
        out_loc = os.path.join(out_dir, out_name+'.npz')

        if not args.force and os.path.isfile(out_loc):
            logging.info('{} already exists. Skipping.'.format(out_name))
            logging.info('Discretize End')
            sys.exit(0)

        dense_spectra =[open_msms_file(ff)[['spectrum','precursor_mz']]
                        for f in args.files
                        for ff in glob.glob(f)]
        file_ids = np.cumsum(np.array([s.spectrum.shape[0] for s in dense_spectra]))
        pmzs = np.concatenate([s.precursor_mz for s in dense_spectra]).tolist()
        dense_spectra = np.concatenate([s.spectrum for s in dense_spectra])

        start = timer()
        S = discretize_spectra(dense_spectra,pmzs=pmzs,bin_width=args.bin_width,
                               intensity_power=args.intensity_power,
                               trim_empty=args.trim,remove_duplicates=args.dedup)
        end = timer()

        S['file_ids'] = file_ids

        write_sparse_msms_file(out_loc, S)

        logging.info('Discretize Time: {} seconds, {} spectra'.format(end-start, S['spec_ids'].max()+1))
        logging.info('Discretize End')

    elif common_ext == '.npz':
        logging.info('Score Start')

        out_name = '_'.join([os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] for f in args.files])

        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = os.path.dirname(os.path.abspath(args.files[0]))
        logging.info('Output to {}'.format(out_dir))

        out_loc = os.path.join(out_dir, out_name)

        if not args.force and os.path.isfile(out_loc):
            logging.info('{} already exists. Skipping.'.format(out_name))
            logging.info('Score End')
            sys.exit(0)

        S1 = open_sparse_msms_file(args.files[0])
        bin_width = S1['bin_width']
        S1_blanks = S1.get('blanks',np.array([]))

        if len(args.files) == 1:
            S2 = S1
            S2_blanks = S1_blanks
        else:
            S2 = open_sparse_msms_file(args.files[1])
            S2_blanks = S2.get('blanks',np.array([]))

            try:
                assert S2['bin_width'] == bin_width
            except AssertionError:
                log.error('Input files have differing bin_width')
                sys.exit(1)

        start = timer()

        S12 = score_sparse_spectra(S1, S2,
                                   mass_diffs=args.mass_diffs,
                                   react_steps=args.react_steps,
                                   tolerance=args.tolerance)

        end = timer()
        logging.info('Score Time: {} seconds'.format(end-start))

        if (args.min_score > 0) or (args.min_matches > 0):
            logging.info('Filtering')
            keep_idx =  S12['mzi'] >= args.min_score
            keep_idx = keep_idx.maximum(S12['mzc'] >= args.min_matches)
            if 'nli' in S12.keys():
                keep_idx = keep_idx.maximum(S12['nli'] >= args.min_score)
            if 'nlc' in S12.keys():
                keep_idx = keep_idx.maximum(S12['nlc'] >= args.min_matches)

            S12['mzi'] = S12['mzi'].multiply(keep_idx).tocoo()
            S12['mzc'] = S12['mzc'].multiply(keep_idx).tocoo()

        else:

            S12['mzi'] = S12['mzi'].tocoo()
            S12['mzc'] = S12['mzc'].tocoo()

        if args.fast_format:
            write_sparse_msms_file(out_loc+'_scores.npz', S12)
        else:

            mzi_df = pd.Series(S12['mzi'].data, name='mzi', index=list(zip(S12['mzi'].col.tolist(), S12['mzi'].row.tolist())))
            mzc_df = pd.Series(S12['mzc'].data, name='mzc', index=list(zip(S12['mzc'].col.tolist(), S12['mzc'].row.tolist())))
            out_df = pd.concat([mzi_df, mzc_df], axis=1)

            out_df.index.names = ['/'.join([str(args.tolerance),
                                            ','.join([str(d) for d in args.mass_diffs]),
                                            str(args.react_steps),
                                            str(args.min_score),
                                            str(args.min_matches)])]

            out_df.to_csv(out_loc+'.tab', index=True, sep='\t', columns = sorted(out_df.columns,key=lambda c:c[::-1])[::-1])

        logging.info('Score End')

    else:
        logging.error('Input files must only be .mgf or .npz')
        sys.exit(1)

if __name__ == '__main__':
    main()