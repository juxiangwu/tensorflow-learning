# -*- coding: utf-8 -*-
'''
转换手写识别文件成jpg
'''

import struct
from PIL import Image
import os


path = 'D:/Develop/DeepLearning/datas/HWDB1.1tst_gnt/'
#global count
for z in range(1241, 1301):
    ff =  path + str(z) + '-c.gnt'
    f = open(ff, 'rb')
    # ifend = f.read(1)
    count = 0
    while f.read(1) != "":
        f.seek(-1, 1)
        #global count
        count += 1
        length_bytes = struct.unpack('<I', f.read(4))[0]
        print
        length_bytes
        tag_code = f.read(2)
        print
        tag_code
        width = struct.unpack('<H', f.read(2))[0]
        print
        width
        height = struct.unpack('<H', f.read(2))[0]
        print
        height

        im = Image.new('RGB', (width, height))
        img_array = im.load()
        # print img_array[0,7]
        for x in range(0, height):
            for y in range(0, width):
                pixel = struct.unpack('<B', f.read(1))[0]
                img_array[y, x] = (pixel, pixel, pixel)

                # print str(count)
        filename = str(count) + '.jpg'
        # filename = '/'+ tag_code + '/' +filename
        print
        filename
        if (os.path.exists(path + str(tag_code))):
            filename = path + str(tag_code) + '/' + filename
            im.save(filename)
            # f.close()
        else:
            os.makedirs(path + str(tag_code))
            filename = path + str(tag_code) + '/' + filename
            im.save(filename)
    f.close()