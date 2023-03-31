# tensorflow2-pointcloud-regularizers
Regularizers including spatial dropout and random rotations for 3D point clouds.

Requirements:
```unix
tensorflow 2.0
tensorflow_graphics
```

Example Usage:
```python
from pointcloud_regularizers import Random_PointSlicing, Random_Voronoi_Dropout, Random_PointRotation

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras

def get_model(numPoints):
    en_yx  =  layers.Input(shape=(numPoints,3),name='yx')

    #x = Random_PointSlicing()(en_yx)
    x = Random_Voronoi_Dropout(numGroups=6, rate=0.5, scales=[1,1,0.1])(en_yx)
    # Add Gaussian noise to the points
    x = layers.GaussianNoise(stddev = (0.1524 / 100.0) )(x)
    # randomly rotate the points around some axis through the origin
    x = Random_PointRotation()(x)

    model = Model(inputs=[en_yx], outputs= [x], name='main_model')
    model.compile()
    return model

def sample_unit(npoints, r, ndim=3):
    # sample points inside of a unit sphere
    vec = tf.random.normal( ( npoints, ndim ) , dtype=np.float32)
    vec /= tf.expand_dims(tf.norm(vec, axis = 1), axis=1)
    vec *= tf.random.uniform( ( npoints, 1 ), maxval=r, dtype=np.float32)
    return vec

tf_to_slice = sample_unit(1000, 1.0, 3)
tf_to_slice = tf.tile(tf.expand_dims(tf_to_slice, axis=0), (10, 1, 1))

model = get_model(tf_to_slice.shape[1])
outp = model(tf_to_slice, training=True)
```
