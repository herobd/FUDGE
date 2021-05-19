import torch
import torch.utils.data
import numpy as np
import json
from skimage import io
from skimage import draw
#import skimage.transform as sktransform
import os
import math
import utils.img_f as img_f
from collections import defaultdict
import random
from random import shuffle
from utils import augmentation
from utils.crop_transform import CropBoxTransform
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints
SKIP=['121','174']


def getDistMask(queryMask,thresh=1000,reverse=True, negative=True):
    dist_transform = img_f.distanceTransform(1-queryMask.astype(np.uint8),cv2.DIST_L2,5)
    dist_transform = np.clip(dist_transform,None,thresh)
    if reverse:
        dist_transform = thresh-dist_transform
    distMask = dist_transform/thresh
    if negative:
        distMask = 2*distMask - 1 #make the mask between -1 and 1, where 1 is on the query and -1 is thresh pixels
    return distMask


def collate(batch):

    ##tic=timeit.default_timer()
    batch_size = len(batch)
    imageNames=[]
    scales=[]
    cropPoint=[]
    imgs = []
    queryMask=[]
    queryClass=[]
    max_h=0
    max_w=0
    bb_sizes=[]
    bb_dim=None
    for b in batch:
        if b is None:
            continue
        imageNames.append(b['imgName'])
        scales.append(b['scale'])
        cropPoint.append(b['cropPoint'])
        imgs.append(b["img"])
        queryMask.append(b['queryMask'])
        queryClass.append(b['queryClass'])
        max_h = max(max_h,b["img"].size(2))
        max_w = max(max_w,b["img"].size(3))
        gt = b['responseBBs']
        if gt is None:
            bb_sizes.append(0)
        else:
            bb_sizes.append(gt.size(1)) 
            bb_dim=gt.size(2)
    if len(imgs) == 0:
        return None

    largest_bb_count = max(bb_sizes)

    ##print(' col channels: {}'.format(len(imgs[0].size())))
    batch_size = len(imgs)

    resized_imgs = []
    resized_queryMask = []
    index=0
    for img in imgs:
        if img.size(2)<max_h or img.size(3)<max_w:
            resized = torch.zeros([1,img.size(1),max_h,max_w]).type(img.type())
            diff_h = max_h-img.size(2)
            pos_r = 0#np.random.randint(0,diff_h+1)
            diff_w = max_w-img.size(3)
            pos_c = 0#np.random.randint(0,diff_w+1)
            #if len(img.size())==3:
                #    resized[:,pos_r:pos_r+img.size(1), pos_c:pos_c+img.size(2)]=img
            #else:
                #    resized[pos_r:pos_r+img.size(1), pos_c:pos_c+img.size(2)]=img
            resized[:,:,pos_r:pos_r+img.size(2), pos_c:pos_c+img.size(3)]=img
            resized_imgs.append(resized)

            if queryMask[index] is not None:
                resized_gt = torch.zeros([1,queryMask[index].size(1),max_h,max_w]).type(queryMask[index].type())
                resized_gt[:,:,pos_r:pos_r+img.size(2), pos_c:pos_c+img.size(3)]=queryMask[index]
                resized_queryMask.append(resized_gt)
        else:
            resized_imgs.append(img)
            if queryMask[index] is not None:
                resized_queryMask.append(queryMask[index])
        index+=1

            

    if largest_bb_count != 0:
        bbs = torch.zeros(batch_size, largest_bb_count, bb_dim)
    else:
        bbs=None
    for i, b in enumerate(batch):
        gt = b['responseBBs']
        if bb_sizes[i] == 0:
            continue
        bbs[i, :bb_sizes[i]] = gt


    imgs = torch.cat(resized_imgs)
    if len(resized_queryMask)==1:
        queryMask = resized_queryMask[0]
    elif len(resized_queryMask)>1:
        queryMask = torch.cat(resized_queryMask)
    else:
        queryMask = None

    ##print('collate: '+str(timeit.default_timer()-tic))
    return {
        'img': imgs,
        'responseBBs': bbs,
        "responseBB_sizes": bb_sizes,
        'queryMask': queryMask,
        'queryClass':queryClass,
        "imgName": imageNames,
        "scale": scales,
        "cropPoint": cropPoint
    }

class FormsBoxPair(torch.utils.data.Dataset):
    """
    Class for reading Forms dataset and creating quer masks from bbbs
    """

    def __getResponseBBList(self,queryId,annotations):
        responseBBList=[]
        for pair in annotations['pairs']: #done already +annotations['samePairs']:
            if queryId in pair:
                if pair[0]==queryId:
                    otherId=pair[1]
                else:
                    otherId=pair[0]
                if otherId in annotations['byId']: #catch for gt error
                    responseBBList.append(annotations['byId'][otherId])
                #if not self.isSkipField(annotations['byId'][otherId]):
                #    poly = np.array(annotations['byId'][otherId]['poly_points']) #self.__getResponseBB(otherId,annotations)  
                #    responseBBList.append(poly)
        return responseBBList


    def __init__(self, dirPath=None, split=None, config=None, instances=None, test=False):
        if split=='valid':
            valid=True
            amountPer=0.25
        else:
            valid=False
        self.cache_resized=False
        if 'augmentation_params' in config:
            self.augmentation_params=config['augmentation_params']
        else:
            self.augmentation_params=None
        if 'no_blanks' in config:
            self.no_blanks = config['no_blanks']
        else:
            self.no_blanks = False
        if 'no_print_fields' in config:
            self.no_print_fields = config['no_print_fields']
        else:
            self.no_print_fields = False
        self.no_graphics =  config['no_graphics'] if 'no_graphics' in config else False
        self.swapCircle = config['swap_circle'] if 'swap_circle' in config else True
        self.onlyFormStuff = config['only_form_stuff'] if 'only_form_stuff' in config else False
        self.only_opposite_pairs = False
        self.color = config['color'] if 'color' in config else True
        self.rotate = config['rotation'] if 'rotation' in config else True
        self.useDistMask = config['use_dist_mask'] if 'use_dist_mask' in config else False
        self.useDoughnutMask = config['use_doughnut_mask'] if 'use_doughnut_mask' in config else False
        self.useVDistMask = config['use_vdist_mask'] if 'use_vdist_mask' in config else False
        self.useHDistMask = config['use_hdist_mask'] if 'use_hdist_mask' in config else False

        self.simple_dataset = config['simple_dataset'] if 'simple_dataset' in config else False
        
        self.rescale_range=config['rescale_range']
        if 'cache_resized_images' in config:
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False
        #self.fixedDetectorCheckpoint = config['detector_checkpoint'] if 'detector_checkpoint' in config else None
        crop_params=config['crop_params'] if 'crop_params' in config else None
        if crop_params is not None:
            self.transform = CropBoxTransform(crop_params,self.rotate)
        else:
            self.transform = None
        if instances is not None:
            self.instances=instances
        else:
            if self.simple_dataset:
                splitFile = 'simple_train_valid_test_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                groupsToUse = json.loads(f.read())[split]
            self.instances=[]
            groupNames = list(groupsToUse.keys())
            groupNames.sort()
            for groupName in groupNames:
                imageNames=groupsToUse[groupName]
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                for imageName in imageNames:
                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    #print(org_path)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
                    annotations=None
                    if os.path.exists(jsonPath):
                        rescale=1.0
                        if self.cache_resized:
                            rescale = self.rescale_range[1]
                            if not os.path.exists(path):
                                org_img = img_f.imread(org_path)
                                if org_img is None:
                                    print('WARNING, could not read {}'.format(org_img))
                                    continue
                                resized = img_f.resize(org_img,(0,0),
                                        fx=self.rescale_range[1],
                                        fy=self.rescale_range[1],
                                        )
                                img_f.imwrite(path,resized)
                        if annotations is None:
                            with open(os.path.join(jsonPath)) as f:
                                annotations = json.loads(f.read())
                            #print(os.path.join(jsonPath))

                            #fix assumptions made in GTing
                            fixAnnotations(self,annotations)

                        #print(path)
                        instancesForImage=[]
                        for id,bb in annotations['byId'].items():
                            if not self.onlyFormStuff or ('paired' in bb and bb['paired']):
                                responseBBList = self.__getResponseBBList(id,annotations)
                                instancesForImage.append({
                                                    'id': id,
                                                    'imagePath': path,
                                                    'imageName': imageName[:imageName.rfind('.')],
                                                    'queryBB': bb,
                                                    'responseBBList': responseBBList,
                                                    'rescaled':rescale,
                                                    #'helperStats': self.__getHelperStats(bbPoints, responseBBList, imH, imW)
                                                })
                        if valid:
                            random.seed(123)
                            shuffle(instancesForImage)
                            self.instances += instancesForImage[:int(amountPer*len(instancesForImage))]
                        else:
                            self.instances += instancesForImage

        


    def __len__(self):
        return len(self.instances)

    def __getitem__(self,index):
        id = self.instances[index]['id']
        imagePath = self.instances[index]['imagePath']
        imageName = self.instances[index]['imageName']
        queryBB = self.instances[index]['queryBB']
        assert(queryBB['type']!='fieldCol')
        queryClassIndex = 0 if queryBB['type'][:4]=='text' else 1
        responseBBList = self.instances[index]['responseBBList']
        rescaled = self.instances[index]['rescaled']
        #xQueryC,yQueryC,reach,x0,y0,x1,y1 = self.instances[index]['helperStats'


        np_img = img_f.imread(imagePath, 1 if self.color else 0)
        if np_img.shape[0]==0:
            print("ERROR, could not open "+imagePath)
            return self.__getitem__((index+1)%self.__len__())
        #Rescale
        scale = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        partial_rescale = scale/rescaled
        np_img = img_f.resize(np_img,(0,0),
                fx=partial_rescale,
                fy=partial_rescale,
                )
        #queryPoly *=scale

        if not self.color:
            np_img = np_img[:,:,None]

        response_bbs = getBBWithPoints(responseBBList,scale)
        query_bb = getBBWithPoints([queryBB],scale)[0,0]

        queryMask = np.zeros([np_img.shape[0],np_img.shape[1]])
        rr, cc = draw.polygon(query_bb[[1,3,5,7]], query_bb[[0,2,4,6]], queryMask.shape)
        queryMask[rr,cc]=1
        #queryMask=queryMask[...,None] #add channel
        masks = [queryMask]
        distMask=None
        if self.useDistMask:
            distMask = getDistMask(queryMask)
            revDistMask = getDistMask(1-queryMask)
            masks.append(distMask)
        if self.useDoughnutMask:
            distMask = getDistMask(queryMask,negative=False)
            smallestDim = min(max(query_bb[[1,3,5,7]])-min(query_bb[[1,3,5,7]]),max(query_bb[[0,2,4,6]])-min(query_bb[[0,2,4,6]]))
            revDistMask = -1*getDistMask(1-queryMask,thresh=smallestDim/2,negative=False)
            distMask = np.where(queryMask,revDistMask,distMask)
            masks.append(distMask)
        if self.useHDistMask:
            if distMask is None:
                distMask = getDistMask(queryMask)
            minY=math.ceil(query_bb[[1,3,5,7]].min())
            maxY=math.floor(query_bb[[1,3,5,7]].max())
            hdistMask = distMask.copy()
            hdistMask[:minY,:]=-1
            hdistMask[maxY:,:]=-1
            masks.append(hdistMask)
        if self.useVDistMask:
            if distMask is None:
                distMask = getDistMask(queryMask)
            minX=math.ceil(query_bb[[0,2,4,6]].min())
            maxX=math.floor(query_bb[[0,2,4,6]].max())
            vdistMask = distMask.copy()
            vdistMask[:,:minX]=-1
            vdistMask[:,maxX:]=-1
            masks.append(vdistMask)

        queryMask = np.stack(masks,axis=2)
        
        #responseMask = np.zeros([image.shape[0],image.shape[1]])
        #for poly in responsePolyList:
        #    rr, cc = draw.polygon(poly[:, 1], poly[:, 0], responseMask.shape)
        #    responseMask[rr,cc]=1

        #imageWithQuery = np.append(1-np_img/128.0,queryMask[:,:,None]),axis=2)
        #sample = self.cropResize(imageWithQuery, responseMask, xQueryC,yQueryC,reach,x0,y0,x1,y1)
        if self.transform is not None:
            out,cropPoint = self.transform({
                "img": np_img,
                "bb_gt": response_bbs,
                "query_bb":query_bb,
                "point_gt": None,
                "pixel_gt": queryMask,
            })
            np_img = out['img']
            response_bbs = out['bb_gt']
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
            np_img = augmentation.apply_tensmeyer_brightness(np_img)
            queryMask = out['pixel_gt']
        else:
            cropPoint=None

        t_response_bbs = convertBBs(response_bbs,self.rotate,2)
        if t_response_bbs is not None and (torch.isnan(t_response_bbs).any() or (float('inf')==t_response_bbs).any()):
            print('nan or inf on {}. response_bbs:{}. response_bbs_trans={}. responseBBList={}'.format(imageName,response_bbs,response_bbs_trans,responseBBList))
            return self.__getitem__((index+1)%len(self.instances))
            #import pdb; pdb.set_trace()

        np_img = np.moveaxis(np_img,2,0)[None,...] #swap channel dim and add batch dim
        t_img = torch.from_numpy(np_img.astype(np.float32))
        t_img = 1.0 - t_img/128.0
        queryMask = np.moveaxis(queryMask,2,0)[None,...] #swap channel dim and add batch dim
        t_queryMask = torch.from_numpy(queryMask.astype(np.float32))
        return {
                'img':t_img,
                'imgName':imageName,
                'queryMask':t_queryMask,
                'queryClass':queryClassIndex,
                'scale': scale,
                'responseBBs':t_response_bbs,
                'cropPoint':cropPoint,
                }

        #def getBBGT(self,useBBs,s):

        #    
        #    bbs = np.empty((1,len(useBBs), 8+8+2), dtype=np.float32) #2x4 corners, 2x4 cross-points, 2 classes
        #    j=0
        #    for bb in useBBs:
        #        tlX = bb['poly_points'][0][0]
        #        tlY = bb['poly_points'][0][1]
        #        trX = bb['poly_points'][1][0]
        #        trY = bb['poly_points'][1][1]
        #        brX = bb['poly_points'][2][0]
        #        brY = bb['poly_points'][2][1]
        #        blX = bb['poly_points'][3][0]
        #        blY = bb['poly_points'][3][1]

        #        field = bb['type'][:4]!='text' 

        #        bbs[:,j,0]=tlX*s
        #        bbs[:,j,1]=tlY*s
        #        bbs[:,j,2]=trX*s
        #        bbs[:,j,3]=trY*s
        #        bbs[:,j,4]=brX*s
        #        bbs[:,j,5]=brY*s
        #        bbs[:,j,6]=blX*s
        #        bbs[:,j,7]=blY*s
        #        #we add these for conveince to crop BBs within window
        #        bbs[:,j,8]=s*(tlX+blX)/2
        #        bbs[:,j,9]=s*(tlY+blY)/2
        #        bbs[:,j,10]=s*(trX+brX)/2
        #        bbs[:,j,11]=s*(trY+brY)/2
        #        bbs[:,j,12]=s*(tlX+trX)/2
        #        bbs[:,j,13]=s*(tlY+trY)/2
        #        bbs[:,j,14]=s*(brX+blX)/2
        #        bbs[:,j,15]=s*(brY+blY)/2
        #        bbs[:,j,16]=1 if not field else 0
        #        bbs[:,j,17]=1 if field else 0    
        #        j+=1
        #    return bbs

