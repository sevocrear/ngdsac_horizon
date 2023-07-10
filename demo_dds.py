import torch
import numpy as np

from torchvision import transforms

from skimage import color
from skimage.draw import line, set_color

from model import Model
import click

from ngdsac import NGDSAC
from loss import Loss

import cv2

from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.domain import DomainParticipant
from dds_data_structures import MainCameraImage, Point2D
from msgs import Points2D

from smoothing import Smooth_KF, Smooth_ParticleFilter
from datetime import datetime

import matplotlib.pyplot as plt

def process_frame(image, imagesize, device, uniform, nn, ngdsac):
    """
    Estimate horizon line for an image and return a visualization.

    image -- 3 dim numpy image tensor
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


def draw_label(img, color, label_text, y_offset=0):
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


def draw_line(data, lX1, lY1, lX2, lY2, clr):
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


def extract_pts(labels, imagesize, padding_top, image_scale):
    """
    Draw disks for a batch of images.

    labels -- line parameters, array shape (Nx2) where
            N is the number of images in the batch
            2 is the number of line parameters (offset,  slope)
    data -- batch of images to draw to
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


@click.command()
@click.option(
    "--topic_in_img", default="camera_images", help="name of the input image topic"
)
@click.option(
    "--topic_out_debug_img",
    default="test_image/front_camera_horizon",
    help="name of the output debug image topic",
)
@click.option(
    "--topic_out_hor_coeff",
    default="front_camera/horizon_coeffs",
    help="name of the horizon Line coeffs topic",
)
@click.option(
    "--model",
    "-m",
    default="models/weights_ngdsac_pretrained.net",
    help="a trained network",
)
@click.option(
    "--capacity",
    "-c",
    type=int,
    default=4,
    help="controls the model capactiy of the network, must match the model to load (multiplicative factor for number of channels)",
)
@click.option(
    "--imagesize",
    "-is",
    type=int,
    default=256,
    help="size of input images to the network, must match the model to load",
)
@click.option(
    "--inlier_thr",
    "-it",
    type=float,
    default=0.05,
    help="threshold used in the soft inlier count, relative to image size",
)
@click.option(
    "--inlier_alpha",
    "-ia",
    type=float,
    default=0.1,
    help="scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)",
)
@click.option(
    "--inlier_beta",
    "-ib",
    type=float,
    default=100.0,
    help="scaling factor within the sigmoid of the soft inlier count",
)
@click.option(
    "--hypotheses",
    "-hyps",
    type=int,
    default=16,
    help="number of line hypotheses sampled for each image",
)
@click.option(
    "--uniform",
    "-u",
    help="disable neural-guidance and sample data points uniformely, use with a DSAC model",
)
@click.option(
    "--score_thr",
    "-st",
    type=float,
    default=0.4,
    help="threshold on soft inlier count for drawing the estimate (range 0-1)",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda",
    help="Device to run the neural network (cuda, cpu)",
)
@click.option("--record/--no-record", default=False)
@click.option("--show/--no-show", default=False)
def main(
    topic_in_img,
    topic_out_debug_img,
    topic_out_hor_coeff,
    model,
    capacity,
    imagesize,
    inlier_thr,
    inlier_alpha,
    inlier_beta,
    hypotheses,
    uniform,
    score_thr,
    device,
    record,
    show,
):
    if record:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(f"output_{datetime.now()}.avi", fourcc, 20.0, (640, 480))
    # setup ng dsac estimator
    ngdsac = NGDSAC(
        hypotheses, inlier_thr, inlier_beta, inlier_alpha, Loss(imagesize), 1
    )

    # Smoother
    smoother = Smooth_KF(im_shape=(imagesize, imagesize))
    smoother_PF = Smooth_ParticleFilter()
    smoother_PF.initialize_particles()
    
    # load network
    nn = Model(capacity)
    nn.load_state_dict(torch.load(model, map_location=torch.device(device)))
    nn.eval()
    nn = nn.to(device)

    domain = DomainParticipant()

    # input topics
    in_topic_img = Topic(domain, topic_in_img, MainCameraImage)

    # output topics
    topic_out_debug_img = Topic(domain, topic_out_debug_img, MainCameraImage)
    topic_out_hor_coeff = Topic(domain, topic_out_hor_coeff, Points2D)

    # Subs-Pubs
    img_reader = DataReader(domain, in_topic_img)

    debug_img_writer = DataWriter(domain, topic_out_debug_img)
    debug_hor_coff_writed = DataWriter(domain, topic_out_hor_coeff)

    cap = cv2.VideoCapture(
        "../../recordings/horizon_imu_2023-07-06-15-41-52/camera_images_clr_log_2023-07-06-15-41-52.mp4"
    )
    # prepare for plot
    plot_dict={"particle":[], "kalman":[], "raw":[], 'length':0}
    
    while True:
        # # Read input data (eulers from imu, image from camera) deleting them from queue
        # image = img_reader.take_one()
        ret, image = cap.read()

        if not (ret):
            break

        #  Pre-process image
        frame = image

        # Process frame
        score, padding_top, image_scale = process_frame(
            frame, imagesize, device, uniform, nn, ngdsac
        )

        # Extract line pts
        if True:  # score > score_thr:
            offset, slope = ngdsac.est_parameters[0]

            # Smooth the line
            smoother.update_horizon(offset, slope, score)
            offset_smooth, slope_smooth = smoother.get_x()
            line_pts_y_smooth = extract_pts(
                np.array([[offset_smooth, slope_smooth]]),
                imagesize=imagesize,
                padding_top=padding_top,
                image_scale=image_scale,
            )[0]

            # Draw line
            draw_line(
                frame,
                0,
                int(line_pts_y_smooth[0]),
                frame.shape[1],
                int(line_pts_y_smooth[1]),
                (0, 255, 0),
            )
            draw_label(frame, (0, 255, 0), "Kalman", y_offset=20)

            line_pts_y = extract_pts(
                np.array([[offset, slope]]),
                imagesize=imagesize,
                padding_top=padding_top,
                image_scale=image_scale,
            )[0]
            # Draw line
            draw_line(
                frame,
                0,
                int(line_pts_y[0]),
                frame.shape[1],
                int(line_pts_y[1]),
                (255, 0, 0),
            )
            draw_label(frame, (255, 0, 0), "raw")

			# Particle Filter
            smoother_PF.predict()
            smoother_PF.update(np.array([[offset, slope]]))
            smoother_PF.resample()
            offset_PF, slope_PF = smoother_PF.get_estimate()
            #Draw
            line_pts_y_PF = extract_pts(
                np.array([[offset_PF, slope_PF]]),
                imagesize=imagesize,
                padding_top=padding_top,
                image_scale=image_scale,
            )[0]
            # Draw line
            draw_line(
                frame,
                0,
                int(line_pts_y_PF[0]),
                frame.shape[1],
                int(line_pts_y_PF[1]),
                (0, 0, 255),
            )
            draw_label(frame, (0, 0, 255), "Particles", y_offset= 40)
            # Prepare horizon line msg
            p1 = Point2D(x=0, y=line_pts_y_smooth[0])
            p2 = Point2D(x=frame.shape[1], y=line_pts_y_smooth[1])
            points = Points2D([p1, p2])

            # Publish all
            debug_hor_coff_writed.write(points)
        debug_img_writer.write(MainCameraImage.from_numpy(frame))

        # write the frame
        if record:
            out.write(cv2.resize(frame, (640, 480), interpolation=1))
        if show:
            cv2.imshow(
                "NGDSAC horizon",
                cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=1),
            )
            if cv2.waitKey(1) == ord("q"):
                break
        
        plot_dict['raw'].append([offset, slope])
        plot_dict['kalman'].append([offset_smooth, slope_smooth])
        plot_dict['particle'].append([offset_PF, slope_PF])
        plot_dict['length'] += 1
    if record:
        out.release()

    _, ax = plt.subplots(nrows=1, ncols=2)
    step = 10
    ax[0].set_title('Offset')
    ax[1].set_title('Slope')
    ax[0].plot(range(plot_dict['length'])[0::step], np.array(plot_dict['raw'])[:,0][0::step], label='raw', alpha = 0.3)
    ax[0].plot(range(plot_dict['length'])[0::step], np.array(plot_dict['kalman'])[:,0][0::step], label = 'kalman', alpha = 0.3)
    ax[0].plot(range(plot_dict['length'])[0::step], np.array(plot_dict['particle'])[:,0][0::step], label = 'particles', alpha = 0.3)
    ax[1].plot(range(plot_dict['length'])[0::step], np.array(plot_dict['raw'])[:,1][0::step], label='raw', alpha = 0.3)
    ax[1].plot(range(plot_dict['length'])[0::step], np.array(plot_dict['kalman'])[:,1][0::step], label = 'kalman', alpha = 0.3)
    ax[1].plot(range(plot_dict['length'])[0::step], np.array(plot_dict['particle'])[:,1][0::step], label = 'particles', alpha = 0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
