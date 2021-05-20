from skimage import color, io
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.img_f as img_f
from utils import util
from utils.util import plotRect
import math
from model.loss import *
from collections import defaultdict
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from model.optimize import optimizeRelationships, optimizeRelationshipsSoft
import json
from utils.forms_annotations import fixAnnotations, getBBInfo
from evaluators.draw_graph import draw_graph



def FUNSDGraphPair_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    useGTGroups = config['gtGroups'] if 'gtGroups' in config else False
    if toEval is None:
        toEval = ['allEdgePred','allEdgeIndexes','allNodePred','allOutputBoxes', 'allPredGroups', 'allEdgePredTypes','allMissedRels','final','final_edgePredTypes','final_missedRels','allBBAlignment']
        if useGTGroups:
            toEval.append('DocStruct')

    draw_verbosity = config['draw_verbosity'] if 'draw_verbosity' in config else 1

    model = trainer.model
    data = instance['img']
    batchSize = data.shape[0]
    assert(batchSize==1)
    targetBoxes = instance['bb_gt']
    imageName = instance['imgName']
    scale = instance['scale']
    target_num_neighbors = instance['num_neighbors']
    if not trainer.model.detector_predNumNeighbors:
        instance['num_neighbors']=None


    trainer.train_hard_detect_limit=99999999999

    trackAtt = config['showAtt'] if 'showAtt' in config else False
    if trackAtt:
        if model.pairer is None:
            for gn in mode.graphnets:
                gn.trackAtt=True
        else:
            trainer.model.pairer.trackAtt=True
    if 'repetitions' in config:
        trainer.model.pairer.repetitions=config['repetitions']
    pretty = config['pretty'] if 'pretty' in config else False
    if 'repetitions' in config:
        trainer.model.pairer.repetitions=config['repetitions']
    useDetections = config['useDetections'] if 'useDetections' in config else False
    if 'useDetect' in config:
        useDetections = config['useDetect']
    confThresh = config['conf_thresh'] if 'conf_thresh' in config else None

    do_saliency_map =  config['saliency'] if 'saliency' in config else False
    do_graph_check_map =  config['graph_check'] if 'graph_check' in config else False


    numClasses=len(trainer.classMap)

    resultsDirName='results'

    if do_saliency_map and outDir is not None:
        if config['cuda']:
            s_data = data.cuda().requires_grad_()
        else:
            s_data = data.requires_grad_()
        trainer.saliency_model.saliency(s_data,(1-data[0].cpu())/2,str(os.path.join(outDir,'{}_saliency_'.format(imageName))))
    if do_graph_check_map and outDir is not None:
        if config['cuda']:
            s_data = data.cuda().requires_grad_()
        else:
            s_data = data.requires_grad_()
        trainer.graph_check_model.check(s_data)

    trainer.use_gt_trans = config['useGTTrans'] if 'useGTTrans' in config else (config['useGTText'] if 'useGTText' in config else False)
    if useDetections:   
        useGT='only_space'
        if type(useDetections) is str:#useDetections=='gt':
            useGT+=useDetections
        losses, log, out = trainer.newRun(instance,useGT,get=toEval)
    else:
        if trainer.mergeAndGroup:
            losses, log, out = trainer.newRun(instance,False,get=toEval)
        else:
            losses, log, out = trainer.run(instance,False)


    if trackAtt:
        if model.pairer is None:
            #liist of graph nets, get all the attention!
            allAttList = [gn.attn for gn in model.graphnets]
        else:
            attList = model.pairer.attn

    if 'allEdgePred' in out:
        allEdgePred = out['allEdgePred']
        allEdgeIndexes = out['allEdgeIndexes']
        allNodePred = out['allNodePred']
        allOutputBoxes = out['allOutputBoxes']
        allPredGroups = out['allPredGroups']
        allEdgePredTypes = out['allEdgePredTypes']
        allMissedRels = out['allMissedRels']
    else:
        allEdgePred = None

    if targetBoxes is not None:
        targetSize=targetBoxes.size(1)
    else:
        targetSize=0

    toRet={}#log
    missing_iter_0 = not any('_0' in k for k in log.keys())
    if allEdgePred is not None:
        for gIter,(edgePred, relIndexes, bbPred, outputBoxes, predGroups, edgePredTypes, missedRels) in enumerate(zip(allEdgePred,allEdgeIndexes,allNodePred,allOutputBoxes,allPredGroups,allEdgePredTypes,allMissedRels)):
            if missing_iter_0:
                gIter+=1



            if trackAtt and (not model.merge_first or gIter>0):
                attList = allAttList[gIter-1 if model.merge_first else gIter]
                data = data.numpy()
                imageO = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0))
                bbs = outputBoxes.numpy()
                for attL,attn in enumerate(attList):
                    image = imageO.copy()
                    if image.shape[2]==1:
                        image = img_f.cvtColor(image,cv2.COLOR_GRAY2RGB)
                    for i in range(len(relIndexes)):
                        
                        ind1 = relIndexes[i][0]
                        ind2 = relIndexes[i][1]
                        x1 = int(round(bbs[ind1,1]))
                        y1 = int(round(bbs[ind1,2]))
                        x2 = int(round(bbs[ind2,1]))
                        y2 = int(round(bbs[ind2,2]))
                        xh = (x1+x2)//2
                        yh = (y1+y2)//2

                        a1 = attn[0,:,ind1,i]
                        a2 = attn[0,:,ind2,i]
                        color1 = (a1[0].item(),a1[1].item(),a1[2].item())
                        color2 = (a2[0].item(),a2[1].item(),a2[2].item())

                        img_f.line(image,(x1,y1),(xh,yh),color1,1)
                        img_f.line(image,(x2,y2),(xh,yh),color2,1)
                    saveName='{}_Att_gI:{}_L:{}.png'.format(imageName,gIter,attL)
                    io.imsave(os.path.join(outDir,saveName),image)




            if outDir is not None:
                if gIter==0 and trainer.model.merge_first:
                    saveName = '{}_gI{}_mergeFirst_recall:{:.2f}_prec:{:.2f}_Fm:{:.2f}'.format(imageName,gIter,float(log['recallMergeFirst_0']),float(log['precMergeFirst_0']),float(log['FmMergeFirst_0']))
                else:
                    saveName = '{}_gI{}_Fms_edge:{:.2f}_rel:{:.2f}_merge:{:.2f}_group:{:.2f}'.format(imageName,gIter,float(log['FmEdge_{}'.format(gIter)]),float(log['FmRel_{}'.format(gIter)]),float(log['FmOverSeg_{}'.format(gIter)]),float(log['FmGroup_{}'.format(gIter)]))

                path = os.path.join(outDir,saveName+'.png')
                draw_graph(outputBoxes,trainer.model.used_threshConf,bbPred.cpu().detach() if bbPred is not None else None,torch.sigmoid(edgePred).cpu().detach(),relIndexes,predGroups,data,edgePredTypes,missedRels,None,targetBoxes,trainer.classMap,path,useTextLines=trainer.model.useCurvedBBs,targetGroups=instance['gt_groups'],targetPairs=instance['gt_groups_adj'],verbosity=draw_verbosity,bbAlignment=out['allBBAlignment'][gIter])

    if outDir is not None:
        if 'final_rel_Fm' in log:
            path = os.path.join(outDir,'{}_final_relFm:{:.2}_r+p:{:.2}+{:.2}_EDFm:{:.2}_r+p:{:.2}+{:.2}.png'.format(imageName,float(log['final_rel_Fm']),float(log['final_rel_recall']),float(log['final_rel_prec']),float(log['final_group_ED_F1']),float(log['final_group_ED_recall']),float(log['final_group_ED_precision'])))
        else:
            path = os.path.join(outDir,'{}_relFm:{:.2}_r+p:{:.2}+{:.2}_EDFm:{:.2}_r+p:{:.2}+{:.2}.png'.format(imageName,float(log['final_rel_BROS_F']),float(log['final_rel_BROS_recall']),float(log['final_rel_BROS_prec']),float(log['ED_F1']),float(log['ED_recall']),float(log['ED_prec'])))

        finalOutputBoxes, finalPredGroups, finalEdgeIndexes, finalBBTrans = out['final']
        draw_graph(finalOutputBoxes,trainer.model.used_threshConf,None,None,finalEdgeIndexes,finalPredGroups,data,out['final_edgePredTypes'],out['final_missedRels'],out['final_missedGroups'],targetBoxes,trainer.classMap,path,bbTrans=finalBBTrans,useTextLines=trainer.model.useCurvedBBs,targetGroups=instance['gt_groups'],targetPairs=instance['gt_groups_adj'],verbosity=draw_verbosity)

    for key in losses.keys():
        losses[key] = losses[key].item()

    if False:
        retData= { 
                   **toRet,
                   **losses,

                 }
    else:
        retData={}
    keep_prefixes=['final_bb_all','final_group','final_rel','prop_rel','DocStruct','F-M','prec@','recall@','bb_Fm','bb_recall','bb_prec','ED_']
    for key,value in log.items():
        if trainer.mergeAndGroup:
            for prefix in keep_prefixes:
                if key.startswith(prefix):
                    if type(value) is np.ndarray:
                        retData[key]={i:[value[i]] for i in range(value.shape[0])}
                    else:
                        retData[key]=[value]
                    break
        else:
            if type(value) is np.ndarray:
                retData[key]={i:[value[i]] for i in range(value.shape[0])}
            else:
                retData[key]=[value]
    return (
             retData,
             None
            )


