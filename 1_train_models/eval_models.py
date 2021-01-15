#!/usr/bin/env python
# coding: utf-8

from fastai import *
from fastai.text import *

def calcPagePrior(train_df, valid_df):
    
    # get label info
    df = pd.concat([train_df, valid_df], sort=False)
    labels = df['label'].tolist()
    composers = sorted(list(set(labels)))
    composer2idx = {c:i for i, c in enumerate(composers)}
    
    # accumulate & normalize
    counts = np.zeros(len(composers))
    for l in labels:
        counts[composer2idx[l]] += 1
    priors = counts / np.sum(counts)

    # format
    priors = torch.from_numpy(priors.reshape((1,-1)))
    
    return priors

def calcAccuracy_fullpage(learner, path, train_df, valid_df, test_df, databunch = None, ensembled = False):
    
    # batch inference
    if databunch is None: # RNNLearner (AWD-LSTM)
        learner.export()
        learner = load_learner(path, test=TextList.from_df(test_df, path, cols='text'))
        try:
            probs, y = learner.get_preds(ds_type=DatasetType.Test, ordered = True) 
        except:
            probs, y = learner.get_preds(ds_type=DatasetType.Test) 
    else: # Generic Learner (RoBERTa, GPT-2)
        learner = Learner(databunch, learner.model)
        probs = learner.get_preds(ds_type=DatasetType.Test)[0].detach().cpu() # not sorted
        sampler = [i for i in databunch.dl(DatasetType.Test).sampler]
        reverse_sampler = np.argsort(sampler)
        probs = probs[reverse_sampler, :]
    
    # ground truth labels
    labels = list(test_df['label'])
    composers = sorted(set(labels))
    composer2idx = {c:i for i, c in enumerate(composers)}
    gt = torch.from_numpy(np.array([composer2idx[l] for l in labels]))
    
    # average if ensembled
    if ensembled:
        boundaries = getPageBoundaries(test_df)
        probs, gt = averageEnsembled(probs, gt, boundaries)
    
    # apply priors
    priors = calcPagePrior(train_df, valid_df)
    probs_with_priors = torch.mul(probs, priors)
    
    # calc accuracy
    acc = accuracy(probs, gt).item()
    acc_with_prior = accuracy(probs_with_priors, gt).item()

    # calc macroF1
    f1 = macroF1(probs, gt)
    f1_with_prior = macroF1(probs_with_priors, gt)
    
    return (acc, acc_with_prior), (f1, f1_with_prior)

def calcAccuracy_fullpage_augmented(learner, path, train_df, valid_df, test_df, databunch = None, ensembled = False):
    
    # batch inference
    if databunch is None: # RNNLearner (AWD-LSTM)
        learner.export()
        learner = load_learner(path, test=TextList.from_df(test_df, path, cols='text'))
        try:
            probs, y = learner.get_preds(ds_type=DatasetType.Test, ordered = True) 
        except:
            probs, y = learner.get_preds(ds_type=DatasetType.Test) 
    else: # Generic Learner (RoBERTa, GPT-2)
        learner = Learner(databunch, learner.model)
        probs = learner.get_preds(ds_type=DatasetType.Test)[0].detach().cpu() # not sorted
        sampler = [i for i in databunch.dl(DatasetType.Test).sampler]
        reverse_sampler = np.argsort(sampler)
        probs = probs[reverse_sampler, :]
    
    # ground truth labels
    labels = list(test_df['label'])
    composers = sorted(set(labels))
    composer2idx = {c:i for i, c in enumerate(composers)}
    gt = torch.from_numpy(np.array([composer2idx[l] for l in labels]))
    
    # average if ensembled
    if ensembled:
        boundaries = getPageBoundaries(test_df)
        probs, gt = averageEnsembled(probs, gt, boundaries)
    
    # apply priors
    priors = calcPagePrior(train_df, valid_df)
    probs_with_priors = torch.mul(probs, priors)
    
    # calc accuracy
    acc = accuracy(probs, gt).item()
    acc_with_prior = accuracy(probs_with_priors, gt).item()

    # calc macroF1
    f1 = macroF1(probs, gt)
    f1_with_prior = macroF1(probs_with_priors, gt)
    
    return (acc, acc_with_prior), (f1, f1_with_prior)

def getPageBoundaries(df):
    queryids = list(df['id'])
    boundaries = []
    for i, qid in enumerate(queryids):
        if qid[-2:] == '_0':
            boundaries.append(i)
    return boundaries

def averageEnsembled(probs, gt, boundaries):
    gt_selected = gt[boundaries]
    accum = torch.zeros((len(boundaries), probs.shape[1]))
    for i, bnd in enumerate(boundaries):
        if i == len(boundaries) - 1:
            accum[i,:] = torch.sum(probs[bnd:,:], axis=0)
        else:
            next_bnd = boundaries[i+1]
            accum[i,:] = torch.sum(probs[bnd:next_bnd,:], axis=0)
    return accum, gt_selected

def macroF1(probs, gt, eps = 1e-9):
    probs = probs.numpy()
    preds = np.argmax(probs, axis=1)
    gt = gt.numpy()
    cm = calcConfusionMatrix(preds, gt, probs.shape[1])
    rec = np.diag(cm) / (np.sum(cm, axis=1) + eps)
    prec = np.diag(cm) / (np.sum(cm, axis=0) + eps)
    F1scores = 2 * prec * rec / (prec + rec + eps)
    macro = np.average(F1scores)
    return macro

def calcConfusionMatrix(preds, gt, nclasses):
    cm = np.zeros((nclasses,nclasses))
    np.add.at(cm, (gt, preds), 1)
    return cm
