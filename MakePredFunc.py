def makePred(path, model, figures=True, ppc=12, cpb=3, n=20 ** 2):
    """
    Classifies a 3D printing image as okay or failed. Only apply the path of a JPEG image type.
    :param path: Path of the image. JPEG only.
    :param model: Path of the input model. .pkl files only.
    :param figures: Show figures. Default is True.
    :param ppc: Pixels per cell of the HOG transformation. Set to what the model used. Default is 12.
    :param cpb: Cells per block of the HOG transformation. Set to what the model used. Default is 3.
    :param n: Image compression dimension. Set to what the model used. Default is 20**2.
    :return: prediction, probability
    """


    import numpy as np
    from PIL import Image
    from skimage.feature import hog
    imgin = Image.open(path)
    img = imgin.convert('L').resize((n, n), Image.ANTIALIAS)
    img_arr = np.array(img, dtype='int32')
    ho, hogimage = hog(img_arr, visualize=True,
                       pixels_per_cell=(ppc, ppc),
                       cells_per_block=(cpb, cpb),
                       block_norm="L2-Hys")
    predin = ho.reshape(1, -1)
    pred = model.predict(predin)
    pred = pred[0]
    prob = model.predict_proba(predin)
    if figures:
        import matplotlib.pyplot as plt
        img_arr2 = np.array(imgin, dtype='int32')
        fig, ax = plt.subplots()
        ax.imshow(img_arr2)
        ax.set_xticks([])
        ax.set_yticks([])
        if pred == 0:
            txt = "Classified as an okay print"
        elif pred == 1:
            txt = "Classified as a failed print"
        fig.text(.5, .975, txt, ha='center')
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.suptitle("Image Processing Steps")
        ax[0].imshow(img_arr)
        ax[0].set_title("Converted Image")
        ax[1].imshow(hogimage)
        ax[1].set_title("HOG Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()
    img.close()
    return pred, prob.max()