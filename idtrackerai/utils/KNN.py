# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking unmarked individuals in large collectives
 
from __future__ import absolute_import, print_function, division
import tensorflow as tf

def kMeansCluster(vector_values, num_clusters, max_num_steps, stop_coeficient = 0.0):
  vectors = tf.constant(vector_values)
  centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                   [0,0],[num_clusters,-1]))
  old_centroids = tf.Variable(tf.zeros([num_clusters,vector_values.shape[1]]))
  centroid_distance = tf.Variable(tf.zeros([num_clusters,vector_values.shape[1]]))
  expanded_vectors = tf.expand_dims(vectors, 0)
  expanded_centroids = tf.expand_dims(centroids, 1)
  distances = tf.reduce_sum(
    tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
  assignments = tf.argmin(distances, 0)

  means = tf.stack([
    tf.reduce_mean(
        tf.boolean_mask(
            vectors, tf.equal(assignments, c)
        ), 0)
    for c in xrange(num_clusters)])

  save_old_centroids = tf.assign(old_centroids, centroids)
  update_centroids = tf.assign(centroids, means)
  init_op = tf.initialize_all_variables()
  performance = tf.assign(centroid_distance, tf.sub(centroids, old_centroids))
  check_stop = tf.reduce_sum(tf.abs(performance))

  with tf.Session() as sess:
    sess.run(init_op)

    for step in xrange(max_num_steps):
      sess.run(save_old_centroids)
      _, centroid_values, assignment_values = sess.run([update_centroids,
                                                        centroids,
                                                        assignments])
      sess.run(check_stop)
      current_stop_coeficient = check_stop.eval()
      if current_stop_coeficient <= stop_coeficient:
        break

    return centroid_values, assignment_values
