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
    def step(self, offset, slope, score):
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
    '''
    Estimate offset, slope by using Particles Filter.
    
    The particle filter algorithm uses a set of particles to represent possible states of the object being tracked. Each particle represents a hypothesis about the object's state, such as its position, orientation, and velocity.

    - The algorithm works by first initializing the particles with some initial values that cover the likely range of the object's state variables. Then, at each time step, the particles are updated based on the measurements from the sensors.
    
    - The update step involves two main parts: prediction and correction.
        - In the prediction step, the particles are moved forward in time according to a motion model that describes how the object is likely to move.
        - In the correction step, each particle's weight is adjusted based on how well its predicted state matches the sensor measurements.
        
    - The weight adjustment is done using Bayes' rule, which allows us to compute the probability of the particle's state given the sensor measurements. The weights are then normalized so that they sum to one, and the particles are resampled with replacement based on their weights. This means that particles with higher weights are more likely to be sampled again, while particles with lower weights may be discarded.
    
    '''
    def __init__(self, im_shape = (256, 256), num_particles = 1000, process_noise = 0.01, measurement_noise = 0.1):
        self.im_shape = im_shape
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.particles = self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self):
        '''
        Initialize N particles with some initial values that cover the likely range of the object's state variables. 
        '''
        particles = np.zeros((self.num_particles, 2))
        particles[:, 0] = np.random.uniform(0, 1, self.num_particles) # 0-1 (range of image)
        particles[:, 1] = np.random.uniform(-2, 2, self.num_particles) # tan
        return particles

    def predict(self):
        '''
        'Move' particles according to the movement or noise (some control), after that,
        run update to update weights according to the given error metric.
        '''
        self.particles += np.random.normal(0, self.process_noise, self.particles.shape)
        # Clip values in possible ranges
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], -2, 2)

    def update(self, measurements, measurement_noise):
        '''
        Update weights according to the given error metric.
        After that, resample.
        '''
        errors = np.abs(self.particles - measurements)
        self.measurement_noise = (1-float(measurement_noise))
        self.weights = np.exp(-np.sum(errors, axis = 1) / self.measurement_noise)
        self.weights /= np.sum(self.weights)

    def resample(self):
        '''
        By assigning equal weights to all particles, we ensure that no particle is favored over the others at the beginning of the filtering process, since we have no other information available about the likely value of the system state. This practice is commonly known as "uniform resampling".

        In summary, filling the weights with ones in the update-resample step is done to ensure that all particles have an equal chance of being selected at the beginning of the filtering process and to ensure proper normalization of the weights computed for the particles.
        
        Resampling is an important step in the particle filter algorithm. It involves selecting a new set of particles based on their weights or likelihoods, in order to replace poorly performing particles with more promising ones.
        '''
        indices = np.random.choice(self.num_particles, self.num_particles, p = self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_x(self):
        '''
        Get estimate.
        '''
        estimate = np.average(self.particles, axis = 0, weights = self.weights)
        return tuple(estimate)
    def step(self, offset, slope, score):
        self.predict()
        self.update(np.array([[offset, slope]]), score)
        self.resample()
    

def select_smoother(smooth_type: str, im_shape=(256, 256)):
    smooth_type = smooth_type.lower()
    if smooth_type == 'none':
        return None
    elif smooth_type == 'kalman':
        return Smooth_KF(im_shape=im_shape)
    else:
        return Smooth_ParticleFilter(im_shape=im_shape)