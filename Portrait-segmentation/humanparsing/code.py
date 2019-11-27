# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
划分训练集和测试集，8:2

"""

import os
root_images_path="/humanparsing/JPEGImages/"

images_name=os.listdir(root_images_path)

#images_name[:int(len(images_name)*0.8)]

save_path="/humanparsing/test.txt"
with open(save_path, 'w', encoding='utf-8') as w:
    for line in images_name[int(len(images_name)*0.8):]:
        #img_name=line.split('/')[-1]
        #img_name=img_name.replace("\n","")
        w.write(line+"\n")






