# determine whether two boxes overlap
def overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
    if ((x1+w1) < x2) or ((x2+w2) < x1):  # one on the left of another
        return False
    if (y1 < (y2-h2)) or (y2 < (y1-h1)):  # one on the upper of another
        return False
    return True

# iou calculation
def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    area_inter = (min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2))
    area_r1 = w1*h1
    area_r2 = w2*h2
    area_union = area_r1 + area_r2 - area_inter
    iou = area_inter / area_union * 100
    return iou


def cascade_detect(img_name):
    img = cv2.imread('{}.jpg'.format(img_name), 0)
    file = open('{}.txt'.format(img_name), 'a')

    print('1')
    # Here load the detector xml file, put it to your working directory
    # cell_cascade = cv2.CascadeClassifier('mydata/cascade.xml')
    cell_cascade = cv2.CascadeClassifier('cascade.xml')
    # cell_cascade = cv2.CascadeClassifier('cells-cascade-20stages.xml')
    print(cell_cascade)
    print('2')
    # Here used cell_cascade to detect cells, with size restriction the detection would be much faster
    cells = cell_cascade.detectMultiScale(img, minSize=(40, 40), maxSize=(55, 55))





    # false positive identification removal
    # check for overlapping bounding boxes
    index1 = 0
    false_identified = set()
    for (x1, y1, w1, h1) in cells:
        index2 = 0
        for (x2, y2, w2, h2) in cells:
            if x1 != x2 and y1 != y2:
                if overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
                    iou_ = iou(x1, y1, w1, h1, x2, y2, w2, h2)
                    # 15 is the threshold for IOU
                    if iou_ > 15: 
                        # using white pixek percentage to determine the correct box
                        im1 = img[y1:y1 + h1, x1:x1 + w1]
                        (thresh1, im1) = cv2.threshold(im1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        im2 = img[y2:y2 + h2, x2:x2 + w2]
                        (thresh2, im2) = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        white1 = cv2.countNonZero(im1)
                        white2 = cv2.countNonZero(im2)
                        total1 = w1 * h2
                        total2 = w2 * h2
                        ratio1 = int(white1 * 100 / float(total1))
                        ratio2 = int(white2 * 100 / float(total2))
                        if ratio1 > ratio2:
                            false_identified.add(index2)
                            # print("{}: {} -> {}/ {} -> {}/ {}".format(index1, index2, iou_, index2, ratio1, ratio2))
                        if ratio2 > ratio1:
                            false_identified.add(index1)
                            # print("{}: {} -> {}/ {} -> {}/ {}".format(index1, index2, iou_, index1, ratio1, ratio2))
            index2 += 1
        index1 += 1

    delete_count = 0
    false_identified = list(false_identified)
    false_identified.sort()
    #false box removal
    for i in false_identified:
        i = i - delete_count
        cells = np.delete(cells, i, axis=0)
        delete_count += 1








    print('3')
    # print(cells)
    # time.sleep(30)
    # Here we draw the result rectangles, (x, y) is the left-top corner coordinate of that triangle
    # We can just use (x, y) to locate each cell
    # w, h are the width and height
    for (x, y, w, h) in cells:
        # print((x,y,w,h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        center_x = x + round(w / 2)
        center_y = y + round(h / 2)
        file.write('{} {} {} {}\n'.format(center_x, center_y, w, h))

    file.close()
    cv2.imwrite('result.jpg', img)