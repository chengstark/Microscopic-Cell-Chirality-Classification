def single_cell_direction(cell_index, frame_index):
    start = time.time()
    if os.stat(txt_path).st_size != 0:
        cells_loc = np.loadtxt(txt_path)
    img_template = get_rotation_template(cell_index, frame_index+1, 0)
    # if frame_index >= 3:
    #     deg_a = rotation_data_recorder[cell_index][frame_index-1]
    #     deg_b = rotation_data_recorder[cell_index][frame_index-2]
    #     img_a = get_rotation_template(cell_index, frame_index-1, deg_a)
    #     img_b = get_rotation_template(cell_index, frame_index-2, deg_b+deg_a)
    degrees = np.arange(-degree_range, degree_range+1, 1)
    score_list = []
    # score_list_a = []
    # score_list_b = []
    # tmp = get_rotation_template_direct_cut(cell_index, frame_index+1, 0)

    # cv2.imwrite('comparison/{}_template_{}_{}.jpg'.format(cell_index, frame_index, 0), img_template)
    # print("{:.4f}".format(ssim(tmp, img_template, data_range=(tmp.max() - tmp.min()))))
    for degree in degrees:
        img_compare = get_rotation_template (cell_index, frame_index, degree)
        # cv2.imwrite('comparison/{}_compare_{}_{}.jpg'.format(cell_index, frame_index, degree), img_compare)
        # ssim_const = ssim(img_template, img_compare, data_range=img_compare.max() - img_compare.min())
        mse_index = mse(img_template, img_compare)
        # if frame_index >= 3:
        #     mse_index_a = mse(img_a, img_compare)
        #     mse_index_b = mse(img_b, img_compare)
        #     score_list_a.append(mse_index_a)
        #     score_list_b.append(mse_index_b)
        # print(mse_index)
        # print('{}: ssim compare {:.4f}'.format(degree, ssim_const))
        # score_list.append(ssim_const)
        score_list.append(mse_index)

    max_value = min(score_list)
    min_index = score_list.index(min(score_list))
    deg_c = min_index - degree_range
    if len(rotation_data_recorder[cell_index]) > 1:
        tmpstdev = statistics.stdev(rotation_data_recorder[cell_index])
    else:
        tmpstdev = 0
    if frame_index >= 3:
        # print(score_list)
        # print(score_list_a)
        # print(score_list_b)
        # min_index_a = score_list_a.index(min(score_list_a))
        # best_degree_a = min_index_a - degree_range
        # min_index_b = score_list_b.index(min(score_list_b))
        # best_degree_b = min_index_b - degree_range
        deg0 = rotation_data_recorder[cell_index][frame_index-2]
        deg1 = rotation_data_recorder[cell_index][frame_index-1]
        tmplist = rotation_data_recorder[cell_index]
        tmplist.append(deg_c)
        stdev_ = statistics.stdev(tmplist)
        if stdev_ > smoothing_stdv_thresh:
            print("Smoothingwith Previous - {} & {}, Current - {}, \nSTDEV: {}".format(deg0, deg1, deg_c, stdev_))
            if deg0*deg1 > 0:
                deg_c = deg_c*0.7 + deg1*0.2 + deg0*0.1

        # if deg_a * deg_b > 0:
        #     print("Smoothing: {}, {}, {}".format(best_degree, best_degree_a, best_degree_b))
        #     best_degree = best_degree*0.7 + best_degree_a*0.2 + best_degree_b*0.1
    # print(score_list)
    # time.sleep(30)
    # print('Max value: {}'.format(max_value))
    # print('Max index: {}'.format(max_index))
    rotation_data_recorder[cell_index].append(deg_c)
    # print(rotation_data_recorder)

    print('Cell {}/{} best degree: {}, STDEV {}'.format(cell_index+1, cells_loc.shape[0], deg_c, tmpstdev))

    # cv2.circle(img, (int(cells_loc[0][0]), int(cells_loc[0][1])), 5, (255, 0, 0), thickness=-1)
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)

    end = time.time()
    time_per_frame.append(end - start)
    # print(end-start)
    return deg_c