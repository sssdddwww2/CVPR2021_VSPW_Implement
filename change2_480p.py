import os
from PIL import Image
from multiprocessing import Pool


DIR='/your/path/to/VSPW'

Target_Dir = '/your/path/to/VSPW_480p'


def change(DIR,video,image):
    img = Image.open(os.path.join(DIR,'data',video,'origin',image))
    w,h = img.size

    if not os.path.exists(os.path.join(Target_Dir,'data',video,'origin')):
        os.makedirs(os.path.join(Target_Dir,'data',video,'origin'))
    img = img.resize((int(480*w/h),480),Image.BILINEAR)
    img.save(os.path.join(Target_Dir,'data',video,'origin',image))

    if os.path.isfile(os.path.join(DIR,'data',video,'mask',image.split('.')[0]+'.png')):
    

        mask = Image.open(os.path.join(DIR,'data',video,'mask',image.split('.')[0]+'.png'))
        mask = mask.resize((int(480*w/h),480),Image.NEAREST)

        if not os.path.exists(os.path.join(Target_Dir,'data',video,'mask')):
            os.makedirs(os.path.join(Target_Dir,'data',video,'mask'))

        mask.save(os.path.join(Target_Dir,'data',video,'mask',image.split('.')[0]+'.png'))
    print('Processing video {} image {}'.format(video,image))

    





#p = Pool(8)
for video in sorted(os.listdir(os.path.join(DIR,'data'))):
    if video[0]=='.':
        continue
    for image in sorted(os.listdir(os.path.join(DIR,'data',video,'origin'))):
        if image[0]=='.':
            continue
#        p.apply_async(change,args=(DIR,video,image,))
        change(DIR,video,image)
#p.close()
#p.join()
print('finish')

