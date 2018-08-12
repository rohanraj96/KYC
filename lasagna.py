import os
import cv2
import glob
import argparse
import datetime
from PIL import Image as Img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte
from wand.image import Image, Color

def check_format(input_file):

    ext = input_file[-3:-1] + input_file[-1]

    if ext == 'pdf':
        file_format = "PDF"
    elif ext == 'jpg':
        file_format = "JPG"
    elif ext == 'png':
        file_format = "PNG"
    else:
        raise ValueError("Please input valid file format: PDF, JPG, PNG")

    return file_format

def pdf2jpg(input_file):

    os.mkdir('pdf/')
    savepath = 'pdf/'

    with Image(filename = input_file, resolution = 300) as img:

        img.background_color = Color("White")
        img.save(filename = savepath)

def read_jpg(path):

    img_dim = (500,500)
    original = mpimg.imread(path)
    resized = cv2.resize(original, img_dim)
    # print("Image shape:", resized.shape)
    if len(resized.shape) == 2:
        img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        img = resized
    
    return img

def read_png(path):

    img_dim = (500,500)
    original = cv2.imread(path)
    resized = cv2.resize(original, img_dim)
    # print("Image shape:", resized.shape)
    if len(resized.shape) == 2:
        img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        img = resized
    
    return img

def process_pdf(img):

    pass
    return processed

def process_jpg(img):

    low_thresh = 210
    upper_thresh = 255
    blur_kernel = (5,5)
    stddev = 0
    canny_minval = 150
    canny_maxval = 220
    nbd_size = 2
    sobel_kernel = 5
    harris_parameter = 0.07
    
    ret, binary = cv2.threshold(img, low_thresh, upper_thresh, cv2.THRESH_BINARY)
    binary = np.uint8(binary)
    blur = cv2.GaussianBlur(binary, blur_kernel, stddev)
    edges = cv2.Canny(np.uint8(np.copy(blur)), canny_minval, canny_maxval)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    corners = cv2.cornerHarris(np.copy(np.uint8(gray)), nbd_size, sobel_kernel, harris_parameter)
    sobelx = cv2.Sobel(corners,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(corners,cv2.CV_64F,0,1,ksize=5)
    
    return corners

def process_png(img):

    low_thresh = 210
    upper_thresh = 255
    blur_kernel = (5,5)
    stddev = 0
    canny_minval = 150
    canny_maxval = 220
    nbd_size = 2
    sobel_kernel = 5
    harris_parameter = 0.07
    
    ret, binary = cv2.threshold(img, low_thresh, upper_thresh, cv2.THRESH_BINARY)
    binary = np.uint8(binary)
    blur = cv2.GaussianBlur(binary, blur_kernel, stddev)
    edges = cv2.Canny(np.uint8(np.copy(blur)), canny_minval, canny_maxval)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    corners = cv2.cornerHarris(np.copy(np.uint8(gray)), nbd_size, sobel_kernel, harris_parameter)
    sobelx = cv2.Sobel(corners,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(corners,cv2.CV_64F,0,1,ksize=5)

    return corners

def find_localminima(histogram):

    return (np.r_[True, histogram[1:] < histogram[:-1]] & np.r_[histogram[:-1] < histogram[1:], True])

def find_endpoints(boolean):

    flag1 = 0
    false_thresh = 40

    for truth in range(len(boolean)):
    
        if (boolean[truth]) & (flag1 == 0): 
            dim1 = truth
            flag1 = 1
            prev_truth = 0
            remaining = len(boolean) - dim1
        if flag1:
            for next_truth in range(remaining):
                index = dim1 + next_truth
                if boolean[index]:
                    dim2 = index
                    prev_truth = 0
                prev_truth += 1
                if prev_truth > false_thresh:
                    break
                
    return dim1, dim2

def build_peaks(blueprint):

    histogram_w = np.sum(blueprint, axis = 1)
    histogram_w[histogram_w > -5] = 0
    boolean_w = find_localminima(histogram_w)
    w1, w2 = find_endpoints(boolean_w)
    histogram_h = np.sum(blueprint, axis = 0)
    histogram_h[histogram_h > -1e-3] = 0
    boolean_h = find_localminima(histogram_h)
    h1, h2 = find_endpoints(boolean_h)

    return [w1, w2, h1, h2]

def crop_img(img, dim):

    w1 = dim[0]
    w2 = dim[1]
    h1 = dim[2]
    h2 = dim[3]

    return (img[w1:w2, h1:h2])

def read_logo(logo_file):

    watermark = mpimg.imread(logo_file)

# We use img_as_ubyte to convert float32 image to uint8. The new BRPL logo w.e.f 7 Aug, 2018 is in float32

    return img_as_ubyte(watermark)

def process_logo(logo_file, dim):

    w1 = dim[0]
    w2 = dim[1]
    h1 = dim[2]
    h2 = dim[3]

    watermark = read_logo(logo_file)
    resized_watermark = cv2.resize(watermark, (h2 - h1, w2 - w1))

    return img_as_ubyte(resized_watermark)

def superimpose_pdf(img, watermark, output_dir, name):

    watermark_intensity = 0.2 # 20%
    kycdoc_intensity = 0.8 # 80%
    if img.shape[2] < 4:
        img = np.dstack([img, np.ones((img.shape[0], img.shape[1]), dtype="uint8") * 255])
    # print(img.shape)
    # print(watermark.shape)
    lasagna = cv2.addWeighted(img, kycdoc_intensity, watermark, watermark_intensity, 0)
    destination = output_dir + 'PDF/' + name + '.jpg'
    # print(destination)
    cv2.imwrite(destination, lasagna)

def superimpose(img, watermark, output_dir):

    watermark_intensity = 0.2 # 20%
    kycdoc_intensity = 0.8 # 80%
    if img.shape[2] < 4:
        img = np.dstack([img, np.ones((img.shape[0], img.shape[1]), dtype="uint8") * 255])
    # print(img.shape)
    # print(watermark.shape)
    lasagna = cv2.addWeighted(img, kycdoc_intensity, watermark, watermark_intensity, 0)
    destination = output_dir + str(datetime.datetime.now()) + '.jpg'
    # print(destination)
    cv2.imwrite(destination, lasagna)

def run(input_file, logo_file, output_dir):

    file_format = check_format(input_file)

    if file_format == 'PDF':

        watermark = read_logo(logo_file)
        # Save as multiple jpg files
        pdf2jpg(input_file)
        # List of all jpg files
        pages = glob.glob('pdf/*')
        os.mkdir(output_dir + 'PDF/')

        for page in pages:

            # print(page)
            img = mpimg.imread(page)
            h,w,a = img.shape
            height = h//2
            quartile = height//2
            width = w
            start = height - quartile
            end = height + quartile
            watermark = cv2.resize(watermark, (width, height))
            white_noise = np.ones((img.shape), dtype = np.uint8) * 255
            white_noise[start:end, :, :] = watermark
            name = page.split('/')[1]
            superimpose_pdf(img, white_noise, output_dir, name)
            os.remove(page)

        os.rmdir('pdf/')

        im_list = []

        for i in os.listdir(output_dir + 'PDF/'):

            # print(output_dir + 'PDF/' + i)
            im = Img.open(output_dir + 'PDF/' + i)
            im_list.append(im)

        pdf_filename = output_dir + str(datetime.datetime.now())
        im1 = im_list.pop(0)
        im1.save(pdf_filename, "PDF", resolution = 100.0, save_all = True, append_images = im_list)
        for i in os.listdir(output_dir + 'PDF/'):
            os.remove(output_dir + 'PDF/' + i)

        os.rmdir(output_dir + 'PDF/')

    elif file_format == 'JPG':

        img = read_jpg(input_file)
        blueprint = process_jpg(img)
        # add rotation code later
        dim = build_peaks(blueprint)
        cropped = crop_img(img, dim)
        watermark = process_logo(logo_file, dim)
        superimpose(cropped, watermark, output_dir)

    elif file_format == 'PNG':

        img = read_png(input_file)
        blueprint = process_png(img)
        # add rotation code later
        dim = build_peaks(blueprint)
        cropped = crop_img(img, dim)
        watermark = process_logo(logo_file, dim)
        superimpose(cropped, watermark, output_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Lasagna')

    parser.add_argument('-i', '--input', type=str, help="Path to documents which have to be watermarked")
    parser.add_argument('-l', '--logo', type=str, help="Path to where watermark logo is saved")
    parser.add_argument('-o', '--output', type=str, help="Path to where the watermarked documents have to be saved")

    args = parser.parse_args()

    list_of_docs = os.listdir(args.input)
    
    for doc in list_of_docs:
        run(args.input + doc, args.logo, args.output)
        os.remove(args.input + doc)


