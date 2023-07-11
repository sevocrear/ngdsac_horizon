import torch
import numpy as np

from torchvision import transforms

from skimage import color

import cv2

def process_frame(image, imagesize, device, uniform, nn, ngdsac) -> tuple():
    """
    Estimate horizon line for an image and return score, image_scale in padding.
    
    Inputs:
    * image -- 3 dim numpy image tensor
    * imagesize (float) - size of the image (NN input)
    * device (str) - device to run NN on (cuda, cpu)
    * uniform (bool) - if use uniform normal prob instead of neural guidance
    * nn (Model) - neural network CNN
    * ngdsac (NGDSAC) - neural guidance model
    
    Outputs:
    * score (float) - confidence about the estimated horizon line
    * image_scale (float) - ratio image was scaled of
    * padding (float) - how much image was padded from top
    """

    # determine image scaling factor
    image_scale = max(image.shape[0], image.shape[1])
    image_scale = imagesize / image_scale

    # convert image to RGB
    if len(image.shape) < 3:
        image = color.gray2rgb(image)

    # store original image dimensions
    src_h = int(image.shape[0] * image_scale)
    src_w = int(image.shape[1] * image_scale)

    # resize and to gray scale
    image = transforms.functional.to_pil_image(image)
    image = transforms.functional.resize(image, (src_h, src_w))
    image = transforms.functional.adjust_saturation(image, 0)
    image = transforms.functional.to_tensor(image)

    # make image square by zero padding
    padding_left = int((imagesize - image.size(2)) / 2)
    padding_right = imagesize - image.size(2) - padding_left
    padding_top = int((imagesize - image.size(1)) / 2)
    padding_bottom = imagesize - image.size(1) - padding_top

    padding = torch.nn.ZeroPad2d(
        (padding_left, padding_right, padding_top, padding_bottom)
    )
    image = padding(image)

    image_src = image.clone().unsqueeze(0)
    image_src = image_src.to(device)

    # normalize image (mean and variance), values estimated offline from HLW training set
    img_mask = image.sum(0) > 0
    image[:, img_mask] -= 0.45
    image[:, img_mask] /= 0.25
    image = image.unsqueeze(0)

    with torch.no_grad():
        # predict data points and neural guidance
        points, log_probs = nn(image)

        if uniform:
            # overwrite neural guidance with uniform sampling probabilities
            log_probs.fill_(1 / log_probs.size(1))
            log_probs = torch.log(log_probs)

        # fit line with NG-DSAC, providing dummy ground truth labels
        ngdsac(
            points,
            log_probs,
            torch.zeros((1, 2)),
            torch.zeros((1)),
            torch.ones((1)),
            torch.ones((1)),
        )

        # normalized inlier score of the estimated line
    score = ngdsac.batch_inliers[0].sum() / points.shape[2]

    image_src = image_src.cpu().permute(0, 2, 3, 1).numpy()  # Torch to Numpy

    return score, padding_top, image_scale


def draw_label(img, color, label_text, y_offset=0) -> None:
    '''
    Draw line label along with it's color in right top angle of the frame.
    
    Inputs:
    * img (NumPy Array) - input image
    * color (tuple) - color of the circle
    * label_text (str) - label of the line
    * y_offset (int) - offset of the text wrt top image edge
    '''
    cv2.putText(
        img,
        label_text,
        (int(img.shape[1] * 0.8), 50 + y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 0),
        2,
    )
    cv2.circle(img, (int(img.shape[1] * 0.8 - 10), 50 + y_offset), 5, color, -1)


def draw_line(data, lX1, lY1, lX2, lY2, clr) -> None:
    """
    Draw a line with the given color and opacity.

    data -- image to draw to
    lX1 -- x value of line segment start point
    lY1 -- y value of line segment start point
    lX2 -- x value of line segment end point
    lY2 -- y value of line segment end point
    clr -- line color, triple of values
    """
    cv2.line(data, (lX1, lY1), (lX2, lY2), clr, thickness=3)


def extract_pts(labels, imagesize, padding_top, image_scale) -> np.array:
    """
    Extract line pts (y-s) from lines' offsets, slopes.
    
    Inputs:
    * labels -- line parameters, array shape (Nx2) where
            N is the number of images in the batch
            2 is the number of line parameters (offset,  slope)
    * image_scale (float) - ratio image was scaled of
    * padding (float) - how much image was padded from top
    * imagesize (int) - size of the input image (NN)
    Output:
    * lines_ys (Numpy Array) - array of lines' y-s (Nx2)
    """

    # number of image in batch
    n = labels.shape[0]
    lines_ys = np.zeros((n, 2))
    for i in range(n):
        # line
        lY1 = int(labels[i, 0] * imagesize)
        lY2 = int(labels[i, 1] * imagesize + labels[i, 0] * imagesize)
        lines_ys[i] = [lY1, lY2]
    # #undo zero padding of inputs
    lines_ys -= padding_top
    lines_ys /= image_scale
    return lines_ys

def draw_line_label(frame, line_pts, color = (0, 0, 0), label = 'raw', y_offset = 0) -> None:
    '''
    Draw line and it's label on the frame.
    
    Inputs:
    * frame - input image
    * line_pts (np.array) - line pts (y-s) (Nx2)
    * color (tuple) - color of the line and label
    * label (str) - label text
    * y_offset - offset wrt top image's edge of the label
    '''
    draw_line(
        frame,
        0,
        int(line_pts[0]),
        frame.shape[1],
        int(line_pts[1]),
        color,
    )
    draw_label(frame, color, label, y_offset)
        
def get_line(smoother = None, offset = 0.5, slope = 0.0, score = 0.5, imagesize = 256, 
             padding_top = 0, image_scale = 1):
    '''
    Apply smoother to the given state [offset, slope] and returns estimated [offset, slope].
    
    Inputs:
    * smoother (Obj) - smoother (None, Kalman Filter, Particle Filter)
    * offset (float) - horizon line offset wrt left-top angle of the frame
    * slope (float) - slope of the horizon line wrt x-axis
    * score (float) - confidence about estimated line
    * image_scale (float) - ratio image was scaled of
    * padding_top (float) - how much image was padded from top
    * imagesize (int) - size of the input image (NN)
    
    Outputs:
    * line_pts_y (np.array) - list of the y-s of the line correspondig x-s (0, img.width).
    * offset (float) - estimated offset
    * slope (float) - estimated slope
    '''
    # Smooth the line with Kalman Filter
    if smoother is not None:
        smoother.step(offset, slope, score)
        offset, slope = smoother.get_x()
    #Draw
    line_pts_y = extract_pts(
        np.array([[offset, slope]]),
        imagesize=imagesize,
        padding_top=padding_top,
        image_scale=image_scale,
    )[0]
    return line_pts_y, offset, slope