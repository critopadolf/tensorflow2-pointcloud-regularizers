from keras import backend
from keras.engine import base_layer
from keras.utils import tf_utils
import numbers
from tensorflow.python.util.tf_export import keras_export    

from tensorflow_graphics.geometry.transformation.axis_angle import rotate
import tensorflow as tf
import numpy as np
import math

class Random_PointRotation(base_layer.BaseRandomLayer):
    """Apply zero-centered Point Cloud Rotation.
    This is useful to fitting tnet
    (random data augmentation).
    As it is a regularization layer, it is only active at training time.
    Call arguments:
      inputs: Input tensor (bs, numpoints, dim).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as input.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.seed = seed

    def call(self, inputs, training=None):
        def noised():     
            #batch_size, num_points, num_dims = tf.shape(inputs)
            inp_shape = tf.shape(inputs)
            
            batch_size = inp_shape[0]
            num_points = inp_shape[1]
            num_dims = inp_shape[2]

            rotation_angles = tf.expand_dims(tf.random.uniform((batch_size,), minval=0.0, maxval=2*np.pi, dtype=inputs.dtype), axis=-1)
            rotation_axes = tf.random.normal((batch_size, 3), dtype=inputs.dtype)
            rotation_axes = rotation_axes / tf.linalg.norm(rotation_axes, axis=-1, keepdims=True)


            rotation_angles = tf.expand_dims(rotation_angles, axis=1)
            rotation_axes = tf.expand_dims(rotation_axes, axis=1)


            rotation_axes = tf.tile(rotation_axes, (1, num_points, 1))

            rotation_angles = tf.tile(rotation_angles, (1, num_points, 1))


            return rotate(
                point= inputs,
                axis= rotation_axes,
                angle= rotation_angles,
                name= 'axis_angle_rotate_pointclouds'
            )

        return backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {"seed": self.seed
                 }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    
class Random_PointSlicing(base_layer.BaseRandomLayer):
    """Apply zero-centered, random Z axis Point Cloud Slicing (Dropout).
    (you could see it as a form of random data augmentation).
    As it is a regularization layer, it is only active at training time.
    Call arguments:
      inputs: Input tensor (bs, numpoints, dim).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as input.
    """

    def __init__(self, min_angle=0.3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.seed = seed
        self.min_angle = min_angle

    def call(self, inputs, training=None):
        def noised():
            #batch_size, num_points, num_dims = tf.shape(inputs)
            inp_shape = tf.shape(inputs)
            
            batch_size = inp_shape[0]
            num_points = inp_shape[1]
            num_dims = inp_shape[2]
               
            start_angle = tf.random.uniform(shape=(batch_size,1), minval = 0.0, maxval = 2*np.pi, dtype=inputs.dtype)

            end_angle = start_angle + self.min_angle + tf.random.uniform(shape=(batch_size,1), minval = 0.0,maxval= (2*np.pi) - self.min_angle, dtype=inputs.dtype)
            # each point cloud gets the same angle
            start_angle = tf.tile(start_angle, (1, num_points) )
            end_angle = tf.tile(end_angle, (1, num_points) )

            # Extract the x and y coordinates of the point cloud
            x = inputs[:, :, 0]
            y = inputs[:, :, 1]


            # Calculate the angle of each point in the point cloud
            angles = tf.math.atan2(y, x)

            angles = tf.where(angles < start_angle, angles + 2*np.pi, angles)

            # Select the points that are within the range of the start and end angles
            mask = tf.cast(((angles >= start_angle) & (angles <= end_angle)), dtype=inputs.dtype)
            mask = tf.tile(tf.expand_dims(mask, axis=-1), (1,1,num_dims) )

            return inputs * mask


        return backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {"seed": self.seed,
                  "min_angle":self.min_angle
                 }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

class Random_Voronoi_Dropout(base_layer.BaseRandomLayer):
    """Apply Random Voronoi Spatial Dropout
    As it is a regularization layer, it is only active at training time.
    Call arguments:
      inputs: Input tensor (bs, numpoints, dim),
              numGroups : number of voronoi centers
              rate : % of voronoi regions to drop
              region_radius : radius of sphere to sample voronoi centers from
              
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
    Input shape:
        # (batch_size, numPoints, numDim) # 
    Output shape:
      Same shape as input.
    """

    def __init__(self, numGroups=4, rate=0.5, region_radius=1.0, scales=[1,1,1], seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid value {rate} received for "
                "`rate`, expected a value between 0 and 1."
            )
        self.supports_masking = True
        self.seed = seed
        self.numGroups = numGroups
        self.rate = rate
        self.region_radius = region_radius
        self.scales = scales

    def call(self, inputs, training=None):
        if isinstance(self.rate, numbers.Real) and self.rate == 0:
            return tf.identity(inputs)
        if training is None:
            training = backend.learning_phase()
            
        
        def sample_unit(npoints, r, dtype, ndim=3):
            # sample points inside of a unit sphere
            vec = tf.random.normal( ( npoints, ndim ) , dtype=dtype)
            vec /= tf.expand_dims(tf.norm(vec, axis = 1), axis=1)
            vec *= tf.random.uniform( ( npoints, 1 ), maxval=r, dtype=dtype)
            vec *= self.scales
            return vec

        def noised():
            #batch_size, num_points, num_dims = tf.shape(inputs)
            inp_shape = tf.shape(inputs)

            batch_size = inp_shape[0]
            num_points = inp_shape[1]
            num_dims = inp_shape[2]
            
            squeeze_dim = batch_size*num_points
            g = sample_unit(self.numGroups, self.region_radius, inputs.dtype, num_dims)
            p = tf.reshape(inputs,(squeeze_dim, num_dims))


            g = tf.tile(tf.expand_dims(g, axis=0), [squeeze_dim, 1, 1])
            p = tf.tile(tf.expand_dims(p, axis=1), [1, self.numGroups, 1])

            dist = tf.norm( (p - g) , axis = -1, ord='euclidean')

            dist_min = tf.math.argmax(-dist, axis=-1)


            numkeep = tf.cast(self.rate * self.numGroups, tf.int64)

            tokeep = tf.cast(tf.where(dist_min >= numkeep, 1, 0), dtype=inputs.dtype)


            tokeep =  tf.tile(tf.expand_dims(tokeep, axis=-1), (1,num_dims)) 
            tokeep =  tf.reshape(tokeep, (batch_size,num_points, num_dims))


            return inputs * tokeep


        return backend.in_train_phase(noised, inputs, training= training )

    def get_config(self):
        config = {"seed": self.seed,
                  "numGroups":self.numGroups,
                  "rate":self.rate,
                  "region_radius":self.region_radius,
                  "scales":self.scales
                 }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
