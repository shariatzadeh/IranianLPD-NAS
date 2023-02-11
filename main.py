input_size = [60, 190, 1]
output_size = [1]

#=============================================
VERBOSITY = 2

def load_dataset(total, ROOT_PATH, OUT_PATH):
    import numpy as np, glob, cv2
    print(ROOT_PATH, input_size)
    trainPath = [ROOT_PATH + "datasets/plates/", ROOT_PATH + "datasets/backgrounds10/"]

    x_ = np.zeros(shape=(total, input_size[0], input_size[1], input_size[2]),dtype='uint8')
    y_ = np.zeros(shape=(total, output_size[0]), dtype='float') #'uint8'
    cnt = 0
    cntim = 0
    import random
    for db in [1]:
        for fn in glob.glob(trainPath[db] + "*.jpg"):
            if cnt < total:
                cntim = cntim + 1
                im = cv2.imread(fn)
                if im is None:
                    print("error: file is empty:", fn)
                imheight = im.shape[0]//2
                imwidth = im.shape[1]//2
                if cntim % 10 == 0:
                    print('|', end='', flush=True)
                for j in range(2):
                    angle = random.uniform(-5, 5)  # [degrees]
                    M = cv2.getRotationMatrix2D(center=(im.shape[1] / 2, im.shape[0] / 2), angle=angle, scale=1)
                    im1 = cv2.warpAffine(src=im, M=M, dsize=(im.shape[1], im.shape[0]))
                    im1 = cv2.resize(src=im1, dsize=(imwidth, imheight))

                    for i in range(5 if db == 0 else 1000):
                        scale = random.uniform(.9, 1.1)
                        cropWidth = int(input_size[1] * scale) - 1
                        aspectRatio = random.uniform(.9, 1.1)
                        cropHeight = int(cropWidth* input_size[0] / input_size[1] * aspectRatio) - 1
                        x = int((imwidth - cropWidth) * random.uniform(.01, 0.99))
                        y = int((imheight - cropHeight) * random.uniform(.01, 0.99))
                        im2 = im1[y:min(y+cropHeight, im1.shape[0]), x:min(x + cropWidth, im1.shape[1])]
                        if len(im2) == 0:
                            print(fn)
                        im2 = cv2.resize(src=im2, dsize=(input_size[1], input_size[0]))
                        im3 = im2[:, :, 1]
                        im3 = (im3 * random.uniform(0.65, 1.1) + random.uniform(-0.15, 0.15))
                        im3[im3 > 255] = 255
                        im3[im3 < 0] = 0
                        rand = random.uniform(0, 1)
                        if rand > .98:
                            imblur = cv2.blur(src=im3, ksize=(3,3))
                            im3 = rand * im3 + (1-rand) * imblur
                        imfg = im2[:, :, 0]
                        #imfg = (imfg > 210)
                        _, imfg = cv2.threshold(imfg, 210, 255, cv2.THRESH_BINARY)
                        #cv2.imwrite('d:/%db.jpg'%cnt, imfg)
                        imfg = cv2.dilate(src=imfg, kernel=np.ones((5, 5), np.uint8), iterations=1)
                        #cv2.imwrite('d:/%dc.jpg'%cnt, imfg)
                        imfg2 = imfg * 1.0
                        imfg2 *= 1.99/255
                        imfg2[imfg2 > 1] = 1
                        #cv2.imwrite('d:/%dd.jpg'%cnt, 255*cv2.resize(imfg2,dsize=(input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST))
                        if db == 1:
                            imfg2 = imfg2 * 0
                        hasntHalfPlate = (imfg2[0, :].mean() < 0.01) and imfg2[imfg2.shape[0]-1, :].mean() == 0 and (imfg2[:, 0].mean() == 0) and imfg2[: , imfg2.shape[1]-1].mean() == 0 #imfg2.mean() > .025/scale

                        if cnt < total and ((imfg2.mean() > 0.001 and hasntHalfPlate) or random.uniform(0, 100) < .03 or db == 1):
                            cv2.imwrite(OUT_PATH + '%05d.jpg' % cnt, im3)
                            x_[cnt, :, :, 0] = im3
                            y_[cnt, 0] = 1 - db
                            cnt = cnt + 1
                            if cnt % 1000 == 0:
                                print(db, end='', flush=True)
    # x is in 0..255, y in 0..1
    return x_, y_


load_dataset(20000, 'd:/platedetection/', 'd:/1/')