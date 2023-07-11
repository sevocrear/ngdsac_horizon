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
from utils import draw_line_label, process_frame, extract_pts, draw_line, draw_label, get_line
from datetime import datetime

import matplotlib.pyplot as plt

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
    smoother_kf = Smooth_KF(im_shape=(imagesize, imagesize))
    smoother_pf = Smooth_ParticleFilter()
    smoother_pf.initialize_particles()
    
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
        # if True:  # score > score_thr:
        offset, slope = ngdsac.est_parameters[0]

        # raw
        line_pts_y, _, _ = get_line(None, offset, slope, score, imagesize, padding_top, image_scale)
        # Kalman Filter
        line_pts_y_kf, offset_kf, slope_kf = get_line(smoother_kf, offset, slope, score, imagesize, padding_top, image_scale)
        # Particle Filter
        line_pts_y_pf, offset_pf, slope_pf = get_line(smoother_pf, offset, slope, score, imagesize, padding_top, image_scale)
        
        # Draw line
        draw_line_label(frame, line_pts_y, (255, 0, 0), "raw")
        # Draw line
        draw_line_label(frame, line_pts_y_kf, (0, 255, 0), "Kalman", y_offset=20)
        # Draw line
        draw_line_label(frame, line_pts_y_pf, (0, 0, 255), "Particles", y_offset= 40)
        
        # Prepare horizon line msg
        p1 = Point2D(x=0, y=line_pts_y_kf[0])
        p2 = Point2D(x=frame.shape[1], y=line_pts_y_kf[1])
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
        plot_dict['kalman'].append([offset_kf, slope_kf])
        plot_dict['particle'].append([offset_pf, slope_pf])
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
