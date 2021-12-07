# https://github.com/yu4u/cutout-random-erasing

import numpy as np

class CutoutRandomErasing:
    eraser = None

    def preprocess(self, image, performedProbability=0.5, minProportionArea=0.02, maxProportionArea=0.2,
        minAspectRatio=0.3, maxAspectRatio=1/0.3, minAreaValue=0, maxAreaValue=1, pixel_level=False, gray_area=False):
        if self.eraser is None:
            self.eraser = self._get_random_eraser(performedProbability, 
                minProportionArea, maxProportionArea,
                minAspectRatio, maxAspectRatio, 
                minAreaValue, maxAreaValue, pixel_level)

        processedImage = self.eraser(image.copy())

        return processedImage

    # p : the probability that random erasing is performed
    # s_l, s_h : minimum / maximum proportion of erased area against input image
    # r_1, r_2 : minimum / maximum aspect ratio of erased area
    # v_l, v_h : minimum / maximum value for erased area
    # pixel_level : pixel-level randomization for erased are
    def _get_random_eraser(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False, gray_area=False):
        def eraser(input_img):
            if input_img.ndim == 3:
                img_h, img_w, img_c = input_img.shape
            elif input_img.ndim == 2:
                img_h, img_w = input_img.shape

            p_1 = np.random.rand()

            if p_1 > p:
                return input_img

            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            if pixel_level:
                if input_img.ndim == 3:
                    c = np.random.uniform(v_l, v_h, (h, w, img_c))
                if input_img.ndim == 2:
                    c = np.random.uniform(v_l, v_h, (h, w))
            else:
                if gray_area:
                    c = np.random.uniform(v_l, v_h)
                elif input_img.ndim == 3:
                    c = (np.random.uniform(v_l, v_h), 
                        np.random.uniform(v_l, v_h), 
                        np.random.uniform(v_l, v_h))

            input_img[top:top + h, left:left + w] = c

            return input_img

        return eraser