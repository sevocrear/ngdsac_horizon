from filterpy.kalman import KalmanFilter
import numpy as np
class Smooth():
    def __init__(self, im_shape = (256, 256)) -> None:
        # Create a Kalman filter with state variables for slope and offset
        kf = KalmanFilter(dim_x=2, dim_z=2)

        # Set the initial state estimate and covariance matrix
        x = np.array([0.0, 0.5])  # initial slope and offset
        kf.x = x
        kf.P = np.diag([0.1, 0.1])  # initial uncertainty

        # Set the transition matrix and process noise
        kf.F = np.array([[1.0, 0.0],
                        [0.0, 1.0]])
        kf.Q = np.diag([0.01, 0.01])

        # Set the observation matrix and measurement noise
        kf.H = np.array([[1.0, 0.0],
                        [0.0, 1.0]])
        kf.R = np.diag([0.1, 0.1])

        self.kf = kf
        self.image_height = im_shape[1]
        self.image_width = im_shape[0]
        
    # Update the filter with new measurements of the horizon line
    def update_horizon(self, offset, slope):
        # convert line_pts to slope and offset
        self.kf.predict()
        self.kf.update([slope, offset])
        
    # Get slope and offset
    def get_x(self, ):
        slope, offset = self.kf.x
        return offset, slope