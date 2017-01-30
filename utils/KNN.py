import tensorflow as tf

def kMeansCluster(vector_values, num_clusters, max_num_steps, stop_coeficient = 0.0):
  vectors = tf.constant(vector_values)
  centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                   [0,0],[num_clusters,-1]))
  old_centroids = tf.Variable(tf.zeros([num_clusters,vector_values.shape[1]]))
  centroid_distance = tf.Variable(tf.zeros([num_clusters,vector_values.shape[1]]))

  expanded_vectors = tf.expand_dims(vectors, 0)
  expanded_centroids = tf.expand_dims(centroids, 1)

  # print 'vectors shape: '+ str(expanded_vectors.get_shape())
  # print 'centroids shape: ' + str(expanded_centroids.get_shape())

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
    #   print "Running step " + str(step)
      sess.run(save_old_centroids)
      _, centroid_values, assignment_values = sess.run([update_centroids,
                                                        centroids,
                                                        assignments])
      sess.run(check_stop)
      current_stop_coeficient = check_stop.eval()
    #   print "coeficient:", current_stop_coeficient
      if current_stop_coeficient <= stop_coeficient:
        break

    return centroid_values, assignment_values
