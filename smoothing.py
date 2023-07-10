from filterpy.kalman import KalmanFilter
import numpy as np
class Smooth_KF():
    def __init__(self, im_shape = (256, 256)) -> None:
        # Create a Kalman filter with state variables for slope and offset
        kf = KalmanFilter(dim_x=2, dim_z=2)

        # Set the initial state estimate and covariance matrix
        x = np.array([0.0, 0.5])  # initial slope and offset
        kf.x = x
        kf.P = np.diag([10., 10.])  # initial uncertainty

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
    def update_horizon(self, offset, slope, score):
        # upd measurement noise according to the score
        self.kf.R = np.diag([1-score]*2)
        # convert line_pts to slope and offset
        self.kf.predict()
        self.kf.update([slope, offset])
    # Get slope and offset
    def get_x(self, ):
        slope, offset = self.kf.x
        return offset, slope

class Smooth_ParticleFilter():
    def __init__(self, im_shape = (256, 256), num_particles = 500, process_noise = 0.1, measurement_noise = 0.1):
        self.im_shape = im_shape
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.particles = self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self):
        particles = np.zeros((self.num_particles, 2))
        particles[:, 0] = np.random.uniform(0, 1, self.num_particles) # 0-1 (range of image)
        particles[:, 1] = np.random.uniform(-2, 2, self.num_particles) # tan
        return particles

    def predict(self):
        self.particles += np.random.normal(0, self.process_noise, self.particles.shape)
        # Clip values in possible ranges
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], -2, 2)

    def update(self, measurements):
        errors = np.abs(self.particles - measurements)
        self.weights = np.exp(-np.sum(errors, axis = 1) / self.measurement_noise)
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p = self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        estimate = np.average(self.particles, axis = 0, weights = self.weights)
        return tuple(estimate)
