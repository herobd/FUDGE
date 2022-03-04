import pytesseract
from pytesseract import Output
from PIL import Image
import sys
import os
import json
import utils.img_f as img_f
import numpy as np



json_path = sys.argv[1]
image_dir = sys.argv[2]
if len(sys.argv)>3:
    extension = sys.argv[3]
    if extension[0]!='.':
        extension = '.'+extension
else:
    extension = '.png'

psm = 7
custom_oem_psm_config = r'--psm {}'.format(psm) #or 13

pad_y=2
pad_x=4
min_height=32

with open(json_path) as f:
    data = json.load(f)

for image_name,preds in data.items():
    img_path = os.path.join(image_dir,image_name+extension)
    tess_out = pytesseract.image_to_data(img_path, output_type=Output.DICT)
    print(tess_out)
    
    image = img_f.imread(os.path.join(image_dir,image_name+extension))
    img_f.imshow('',image)
    img_f.show()
    for entity in preds['entities']:
        for line in entity['lines']:
            (tlX,tlY),(trX,trY),(brX,brY),(blX,blY) = line['corners']
            assert tlX==blX and tlY==trY and brX==trX and blY==brY #axis aligned
            #line_img = getPoly(image,line['corners'])
            #line_img = image[tlY-pad_y:blY+pad_y,tlX-pad_x:trX+pad_x]

            ##print(line_img.shape)
            #if line_img.shape[0]<min_height:
            #    scale = min_height/line_img.shape[0]
            #    line_img = img_f.resize(line_img,[0],fx=scale,fy=scale)
            #    line_img = line_img.astype(np.uint8)
            ##print(line_img.shape)


            #line_img_PIL = Image.fromarray(line_img)
            #try:
            #    text = pytesseract.image_to_string(line_img_PIL,config=custom_oem_psm_config)
            #except pytesseract.pytesseract.TesseractError:
            #    custom_oem_psm_config += r' --tessdata-dir "$HOME/local/share/tessdata"'
            #    text = pytesseract.image_to_string(line_img_PIL,config=custom_oem_psm_config)
            #text = text.strip()
            #print(text)
            #img_f.imshow('',line_img)
            #img_f.show()
            #line['tesseract_text'] = text[:-2]




            

with open(json_path,'w') as f:
    json.dump(data,f)
