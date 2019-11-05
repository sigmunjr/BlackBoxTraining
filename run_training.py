import sys, os
import slim
sys.path.append(os.path.dirname(slim.__file__))
from object_detection.model_main import main as run_training

import tensorflow as tf


tf.app.run(run_training)

del run_training
from object_detection.export_inference_graph import main as export_graph

print('FERDIG MED TRENING')
FLAGS.input_type='image_tensor'
FLAGS.trained_checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.model_dir)
FLAGS.output_directory=FLAGS.model_dir
tf.app.run(export_graph)

#run_training(None)
