
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage



def left_ventricle(img_data1, slicenumber, thresholds, seeds, closing_holes, dilation_iters):
    def distance(a, b):
        return np.abs(a - b)

    def region_growing_average(img, img_t, tolerance, seed):
        x = seed[0];
        y = seed[1]
        img_t[x, y] = 1
        avg = np.mean(img[np.where(img_t == 1)])
        # check matrix border and conquering criterion for the 4-neigbourhood
        if (y + 1 < img.shape[1] and img_t[x, y + 1] == 0 and distance(avg, img[x, y + 1]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x, y + 1])

        if (y - 1 >= 0 and img_t[x, y - 1] == 0 and distance(avg, img[x, y - 1]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x, y - 1])

        if (x + 1 < img.shape[0] and img_t[x + 1, y] == 0 and distance(avg, img[x + 1, y]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x + 1, y])

        if (x - 1 >= 0 and img_t[x - 1, y] == 0 and distance(avg, img[x - 1, y]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x - 1, y])

    image_0 = []

    for i in range(0, 24):
        image_0.append(img_data1[slicenumber, :, :, i])

    final_ven_area_imgs = []

    def segment_left_ven_area(main_image, thresholds, seed_point, closing_holes):
        def distance(a, b):
            return np.abs(a - b)

        img = main_image.copy();
        img_t = np.zeros(main_image.shape);
        tolerance = thresholds
        seed = seed_point
        x = seed[0];
        y = seed[1]
        img_t[x, y] = 1
        avg = np.mean(img[np.where(img_t == 1)])
        # check matrix border and conquering criterion for the 4-neigbourhood
        if (y + 1 < img.shape[1] and img_t[x, y + 1] == 0 and distance(avg, img[x, y + 1]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x, y + 1])

        if (y - 1 >= 0 and img_t[x, y - 1] == 0 and distance(avg, img[x, y - 1]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x, y - 1])

        if (x + 1 < img.shape[0] and img_t[x + 1, y] == 0 and distance(avg, img[x + 1, y]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x + 1, y])

        if (x - 1 >= 0 and img_t[x - 1, y] == 0 and distance(avg, img[x - 1, y]) <= tolerance):
            region_growing_average(img, img_t, tolerance, [x - 1, y])

        output_image = ndimage.binary_fill_holes(img_t, structure=np.ones(closing_holes)).astype(int)
        final_ven_area_imgs.append(np.array(output_image, dtype='uint8'))

        #         plt.figure(figsize=(10,10))
        #         plt.subplot(2,4,1); plt.imshow(main_image,vmax=2500); plt.title('original image')
        #         plt.subplot(2,4,2); plt.imshow(img_t); plt.title('Segmented')
        #         plt.subplot(2,4,3); plt.imshow(ndimage.binary_fill_holes(img_t, structure=np.ones(closing_holes)).astype(int)); plt.title('closing holes')
        #         plt.subplot(2,4,4); plt.imshow(output_image); plt.title('fully Segmented')
        #         plt.subplots_adjust(left=0.1,
        #                             bottom=0.1,
        #                              right=1,
        #                             top=0.9,
        #                             wspace=0.5,
        #                             hspace=0.5)
        #         plt.show()
        return output_image

    for i in range(0, 24):
        #         print('image: ',i)
        segment_left_ven_area(image_0[i], thresholds[i], seeds[0], closing_holes[0])

    final_ven_area_imgs = np.array(final_ven_area_imgs, dtype='uint8')
    #     print(final_ven_area_imgs.shape)

    from skimage.morphology import dilation

    final_outer_ring = []

    def outer_ring_seg(main_image, inner_ring_mask, dilation_iters, img_no):
        dil = inner_ring_mask
        for i in range(dilation_iters):
            dil = dilation(dil)
        dilated_image = dil * main_image
        ring_image = dilated_image - (inner_ring_mask * main_image)
        #   ring_image = np.where(ring_image>1,ring_image,0)

        ring_image_mask = np.where(ring_image < 1, ring_image, 1)

        #         plt.figure(figsize=(10,10))
        #         plt.subplot(2,4,1); plt.imshow(main_image,vmax=1000); plt.title('original_'+str(img_no))
        #         plt.subplot(2,4,2); plt.imshow(dilated_image,vmax=1000); plt.title('dilated image_'+str(img_no))
        #         plt.subplot(2,4,3); plt.imshow(ring_image,vmax=1000); plt.title('Segmented outer ring_'+str(img_no))
        #         plt.subplot(2,4,4); plt.imshow(ring_image_mask); plt.title('Segmented outer ring mask_'+str(img_no))
        #         plt.subplots_adjust(left=0.1,
        #                             bottom=0.1,
        #                              right=1,
        #                             top=0.9,
        #                             wspace=0.5,
        #                             hspace=0.5)
        #         plt.show()
        final_outer_ring.append(ring_image_mask)
        return ring_image_mask, ring_image

    for i in range(0, 24):
        outer_ring_seg(image_0[i], final_ven_area_imgs[i], dilation_iters[i], i)

    final_outer_ring = np.array(final_outer_ring, dtype='uint8')

    final_area_left_ven_series = []

    def area_of_left_ven(img1):
        area = np.bincount(img1.flatten())[1]
        final_area_left_ven_series.append(area)
        return area

    for i in range(0, 23):
        area_of_left_ven(final_ven_area_imgs[i])

    #     print(final_area_left_ven_series)

    #     return final_ven_area_imgs,final_outer_ring

    import math

    final_thickeness_series = []

    def thickness_cal(left_ventricle_mask, outer_ring_mask):
        def dilation_img(img, iters):
            dil = img
            for i in range(iters):
                dil = dilation(dil)
            return dil

        img1 = left_ventricle_mask
        img2 = outer_ring_mask
        outer_ring_full = img1 + img2
        dilated_image = dilation_img(img1 + img2, 1)
        contour_img1 = dilated_image - outer_ring_full
        contour_img2 = dilation_img(img1, 1) - img1

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 3, 1);
        plt.imshow(contour_img1);
        plt.title('Contour of outer ring')
        plt.subplot(2, 3, 2);
        plt.imshow(contour_img2);
        plt.title('Contour of inner left ventricle')
        plt.subplot(2, 3, 3);
        plt.imshow(contour_img1 + contour_img2);
        plt.title('Contours of whole')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.5)
        plt.show()

        locations_img2 = []
        for x, x1 in enumerate(contour_img2):
            for y, y1 in enumerate(x1):
                if contour_img1[x][y] == 1:
                    locations_img2.append([x, y])

        locations_img1 = []
        for x, x1 in enumerate(contour_img1):
            for y, y1 in enumerate(x1):
                if contour_img2[x][y] == 1:
                    locations_img1.append([x, y])

        final_thickness_values = []
        minn = {}
        for i in locations_img1:
            for j in locations_img2:
                dx2 = (i[0] - j[0]) ** 2  # (200-10)^2
                dy2 = (i[1] - j[1]) ** 2
                distance = math.sqrt(dx2 + dy2)
                try:
                    if distance < minn[str(i)]:
                        minn[str(i)] = distance
                except:
                    minn[str(i)] = distance
        final_thickness_outer_ring = np.mean(list(minn.values()))

        print('Average thickness of the ring is:', final_thickness_outer_ring)
        final_thickeness_series.append(final_thickness_outer_ring)
        return final_thickness_outer_ring

    for i in range(0, 23):
        thickness_cal(final_ven_area_imgs[i], final_outer_ring[i])

    return final_thickeness_series, final_area_left_ven_series