from . import graph
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil


#NFEATURES = 28**2
#NCLASSES = 10
BN_EPSILON = 0.001

# Common methods for all models


class base_model(object):
    
    def __init__(self):
        self.regularizers = []
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None, CAM=False):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        
        sess = self._get_session(sess)
        
        if CAM:
            CAM_map = np.empty([size, self.L_post.shape[0], self.M[-1].astype('int')])
            print('Size CAM_map = {0} x {1} x {2}'.format(size, self.L_post.shape[0], self.M[-1].astype('int')))
       
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_length = end - begin
            
            ### Added by Pierre
            if self.V:
                batch_data = np.zeros((self.batch_size, data.shape[1], self.V))
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1]))
            ###
            
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            # feed_dict = {self.ph_data: batch_data, self.ph_cam: CAM,self.ph_training: 0}
                        
            if CAM:
                feed_dict = {self.ph_data: batch_data, self.ph_training: 0, self.ph_cam: CAM}                          #change
            else:
                feed_dict = {self.ph_data: batch_data, self.ph_training: 0}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                #loss += (batch_loss * batch_length) / size
                loss += sklearn.metrics.mean_squared_error(batch_labels[:end-begin], batch_pred[:end-begin]) * batch_length / size
            else:
                if CAM:
                    #compute CAM map
                    batch_pred, CAM_pred = sess.run([self.op_prediction, self.op_map], feed_dict)
                    if len(CAM_pred.shape) < 3:
                        CAM_pred = np.expand_dims(CAM_pred, 2)
                    TT1, TT2, TT3 = CAM_pred.shape
                    print('Size CAM_pred = {0} x {1} x {2}'.format(TT1, TT2, TT3))
                    print('Begin = {0} ; End = {1}'.format(begin, end))
                    length_batch = end - begin
                    CAM_map[begin:end,:,:] = CAM_pred[0:length_batch]
                else:
                    batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            
        if labels is not None:
            return predictions, loss
        else:
            if CAM:
                return predictions, CAM_map
            else:
                return predictions
        
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        if self.task == 'classification':
            ncorrects = sum(predictions == labels)
            accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
            f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')

            # Added by Emanuel (f-strings introduced into newer versions of Python)
            string = f'accuracy: {accuracy:.2f} ({ncorrects} / \
                    {len(labels)}), f1 (weighted): {f1:.2f}, loss: {loss:.2e}'
            # string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
            #         accuracy, ncorrects, len(labels), f1, loss)
        else:
            accuracy = sklearn.metrics.mean_absolute_error(labels, predictions)
            MSE = sklearn.metrics.mean_squared_error(labels, predictions)
            f1 = sklearn.metrics.r2_score(labels, predictions)

            # Added by Emanuel (f-strings introduced into newer versions of Python)
            string = f'MAE: {accuracy:.2f}, MSE: {MSE:.2f}, R2 score: {f1:.2f}, loss: {loss:.2e}'
            # string = 'MAE: {:.2f}, MSE: {:.2f}, R2 score: {:.2f}, loss: {:.2e}'.format(
            #         accuracy, MSE, f1, loss)
        if sess is None:
            time_run = time.process_time() - t_process
            time_wall = time.time() - t_wall
            string += f'\ntime: {time_run:.0f}s (wall {time_wall:.0f}s)'
            # string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, accuracy, f1, loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()

        sess = tf.Session(graph=self.graph)
#         shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
#         shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
#         os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        path_best_model = os.path.join(self._get_path('checkpoints'), 'best_model')

        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        R2s = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = np.copy(train_data[idx,:]), np.copy(train_labels[idx])
            ##
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
                
            ### Added by Pierre
            # Pass parameters for data augmentation
            r_xyz = (np.random.random([self.batch_size, 3]) - 0.5) * self.max_angle_radian
            h_factor = 1.0 - (np.random.random([self.batch_size, 1]) - 0.5) * self.mag_zoom
            do_augment = np.random.random([self.batch_size, 1]) < self.prob_augment
            aug_params = np.concatenate([r_xyz, h_factor, do_augment], axis=1)
            #print('Aug params: {}'.format(aug_params[0,:]))
            #print('batch_data before augmentation: {}'.format(batch_data[0,0,:]))
            batch_data = self.augment_data(batch_data, aug_params)
            #print('batch_data after augmentation: {}'.format(batch_data[0,0,:]))
            ###
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: 1}
            # Memory monitoring
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict=feed_dict, options=run_options)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                
                
                if self.task == 'classification':
                    if accuracy > self.best_accuracy:
                        self.op_saver.save(sess, path_best_model)
                        string = string + ' * '
                        self.best_accuracy = accuracy
                else:
                    R2s.append(f1)
                    if accuracy < self.best_accuracy:
                        self.op_saver.save(sess, path_best_model)
                        string = string + ' * '
                        self.best_accuracy = accuracy
                    
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        if self.task == 'classification':
            print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        else:
            print('validation accuracy: minimum MAE = {:.2f}, peak R2 = {:.2f}'.format(min(accuracies), max(R2s)))
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def build_graph(self, M_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                ### Added by Pierre
                if self.V:
                    self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, self.V), 'data')
                else:
                    self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                ###
                if self.task == 'classification':
                    self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                else:
                    self.ph_labels = tf.placeholder(tf.float32, (self.batch_size), 'labels')
                    
                self.ph_training = tf.placeholder(tf.int8, (), 'is_training')
                self.ph_cam     = tf.placeholder(tf.bool, (), 'cam')

            # Model.
            op_logits, self.op_map = self.inference(self.ph_data, self.ph_cam,self.ph_training)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            #self.op_saver = tf.train.Saver(max_to_keep=5)
            self.op_saver = tf.train.Saver(max_to_keep=10000)
        
        self.graph.finalize()
    
    def inference(self, data, do_cam, is_training):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits, cam = self._inference(data, do_cam, is_training)
        return logits, cam

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            if self.task == 'classification':
                prediction = tf.argmax(logits, axis=1)
            else:
                prediction = tf.squeeze(logits, axis=1)
                
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                if self.task == 'classification':
                    labels = tf.to_int64(labels)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    cross_entropy = tf.reduce_mean(cross_entropy)
                    tr_accuracy = tf.argmax(logits, axis=1)
                    tr_accuracy = tf.equal(tr_accuracy, labels)
                    tr_accuracy = tf.to_float(tr_accuracy)
                    tr_accuracy = tf.reduce_mean(tr_accuracy)
                else:
                    logits = tf.squeeze(logits, axis=1)
                    #cross_entropy = tf.losses.huber_loss(predictions=logits, labels=labels, delta=0.01)
                    cross_entropy = tf.losses.mean_squared_error(predictions=logits, labels=labels)
                    #cross_entropy = tf.losses.absolute_difference(predictions=logits, labels=labels)
                    #tr_accuracy = tf.sqrt(cross_entropy)
                    #tr_accuracy = tf.losses.mean_squared_error(predictions=logits, labels=labels)
                    tr_accuracy = tf.losses.absolute_difference(predictions=logits, labels=labels)
                    #tr_accuracy = tf.reduce_mean(tr_accuracy)
                    #total_error = tf.reduce_sum(tf.square(tf.subtract(labels, tf.reduce_mean(labels))))
                    #unexplained_error = tf.reduce_sum(tf.square(tf.subtract(logits, labels)))
                    #cross_entropy = tf.subtract(tf.divide(unexplained_error, total_error), 1)
                    
                    
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            #loss = tf.identity(cross_entropy)
            
            # Summaries for TensorBoard.
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss, tr_accuracy])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                tf.summary.scalar('loss/avg/tr_accuracy', averages.average(tr_accuracy))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
#                 optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                #optimizer = tf.train.AdagradOptimizer(learning_rate)
#                 optimizer = tf.train.AdadeltaOptimizer(1.0)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            ## Added by Pierre -- Gradient clipping
            #grad_noise = tf.add(1.0, tf.to_float(global_step))
            #grad_noise = tf.pow(grad_noise, 0.5)
            #grad_noise = tf.divide(0.01, grad_noise)
            #tf.summary.scalar('gradient_noise', grad_noise)
            grad, var = zip(*grads)
            grad_norm = tf.global_norm(grad)
            #grad, _ = tf.clip_by_global_norm(grad, 0.5)
            #grads = zip(grad, var)
            #grads = [(tf.add(grad, tf.random_normal(tf.shape(grad),stddev=grad_noise)), var) for grad, var in grads]
            #grads = [(grad, var) for grad, var in grads]
            ##
            
            tf.summary.scalar('gradient_norm', grad_norm)
            
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients] + update_ops):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        ### Added by Emanuel ###
        full_path = f"{self.dir_name}/{folder}"
        if not os.path.isdir(full_path):
            os.mkdir(full_path)
        return full_path
        ###
        
#         path = os.path.dirname(os.path.realpath(__file__))
#         return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            # ORIGINAL
            #filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            #self.op_saver.restore(sess, filename)
            # MODIFIED PB
            path_best_model = os.path.join(self._get_path('checkpoints'), 'best_model')
            self.op_saver.restore(sess, path_best_model)
        return sess

    def _weight_variable(self, shape, regularization=True, name='weights'):
        #with tf.device('/cpu:0'):
        #initial = tf.truncated_normal_initializer(0, 0.01)
#         initial = tf.random_uniform_initializer(-0.2, 0.2)
        #initial = tf.contrib.layers.xavier_initializer()
        
        ### Added by Emanuel ###
        initial = tf.initializers.glorot_uniform(seed=42)
        ###
        
        var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        with tf.device('/cpu:0'):
#             initial = tf.constant_initializer(0.0)
            
            ### Added by Emanuel ###
            initial = tf.initializers.glorot_uniform(seed=42)
            ###
        
            var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var
    
    def augment_data(self, x, aug_params):
        """
        Apply linear transformation + random normal noise
        
        input of size batch_size x N_nodes x 3 (XYZ coordinates)
        or            bacth_size x N_nodes x 6 (grey and white surfaces)
        """
        B, N, C = x.shape
        
        for i in range(B):
            # Extract and reshape a batch
            slice_oi = x[i,:,:]
            slice_oi = np.transpose(slice_oi) # C x N
            
            # Collect random params
            t_x = aug_params[i,0]
            t_y = aug_params[i,1]
            t_z = aug_params[i,2]
            h_factor = aug_params[i,3]
            do_augmentation = aug_params[i,4]
            
            # Create transformation matrix
            r_x = np.matrix([[1, 0, 0], [0, np.cos(t_x), -np.sin(t_x)], [0, np.sin(t_x), np.cos(t_x)]])
            r_y = np.matrix([[np.cos(t_y), 0, np.sin(t_y)], [0, 1, 0], [-np.sin(t_y), 0, np.cos(t_y)]])
            r_z = np.matrix([[np.cos(t_z), -np.sin(t_z), 0], [np.sin(t_z), np.cos(t_z), 0], [0, 0, 1]])
            H   = np.matrix([[h_factor, 0, 0], [0, h_factor, 0], [0, 0, h_factor]])
            
            if do_augmentation > 0.5:
                x_mat = np.matmul(r_x, r_y)
                x_mat = np.matmul(x_mat, r_z)
                x_mat = np.matmul(x_mat, H)
            else:
                x_mat = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            # Rx = [1 0 0; 0 cos(t_x) -sin(t_x); 0 sin(t_x) cos(t_x)];
            # Ry = [cos(t_y) 0 sin(t_y); 0 1 0; -sin(t_y) 0 cos(t_y)];
            # Rz = [cos(t_z) -sin(t_z) 0; sin(t_z) cos(t_z) 0; 0 0 1];
            # H = [h_factor 0 0; 0 h_factor 0; 0 0 h_factor];
            
            # Apply transformation matrix
            if C == 3:
                slice_oi = np.matmul(x_mat, slice_oi)
            elif C == 6:
                slice_1 = slice_oi[0:3,:]
                slice_2 = slice_oi[3:6,:]
                slice_1 = np.matmul(x_mat, slice_1)
                slice_2 = np.matmul(x_mat, slice_2)
                slice_oi = np.concatenate((slice_1, slice_2), axis = 0)
            
            #### Added by Emanuel ####
            elif C == 9:
                slice_1 = slice_oi[0:3,:]
                slice_2 = slice_oi[3:6,:]
                slice_3 = slice_oi[6:9,:]
                
                slice_1 = np.matmul(x_mat, slice_1)
                slice_2 = np.matmul(x_mat, slice_2)
                slice_3 = np.matmul(x_mat, slice_3)
                
                slice_oi = np.concatenate((slice_1, slice_2, slice_3), axis = 0)
            ####
            
            # Apply dropout
            #if do_augmentation == 1:
            #    do_filter = np.random.random([1, N]) < 0.9
            #    slice_oi = np.multiply(slice_oi, do_filter)
            
            # Add noise
            #if do_augmentation == 1:
            #    slice_oi += np.random.normal(loc=0, scale=self.rand_noise, size=slice_oi.shape)
            slice_oi += np.random.normal(loc=0, scale=self.rand_noise, size=slice_oi.shape)
            
            # Transpose slice_oi back
            slice_oi = np.transpose(slice_oi) # N x C
            
            # Replace slice in X
            x[i,:,:] = slice_oi
        
        return x


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F:   Number of features per ResNet layer.
              Default = 20
        K:   Polynomial order per ResNet layer
              Default = 6
        p:   Pooling size for each ResNet layer
             Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
             Beware to have coarsened enough.
              Default = 2
        B:   Number of ResNet blocks
              Default = 10
        F_l: Number of features for the post ResNet layer
              Default = 32
        ### Added by Pierre
        V: Length of vertex-wise vectors
        ###

    L: List of Graph Laplacians. Size M x M. One per coarsening level.
    
    The following are choices of implementation for various blocks.
        pool: pooling type. Can be mpool1 or apool1
            Default = apool1
        
    Task parameter:
        task:          Must be either 'classification' or 'regression'
    
    Training parameters:
        num_epochs:    Number of training epochs.
                        Default = 20
        learning_rate: Initial learning rate.
                        Default = 0.1
        decay_rate:    Base of exponential decay. No decay with 1.
                        Default = 0.95
        decay_steps:   Number of steps after which the learning rate decays.
                        Default = None
        momentum:      Momentum. 0 for ADAM optimizer.
                        Default = 0
        dropout:       Dropout for the FC layer
                        Default = 0.5

    Regularization parameters:
        regularization: L2 regularizations of weights
                         Default = 0
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
                         Default = 100
        eval_frequency: Number of steps between evaluations.
                         Default = 200

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, L, M, F=20, K=6, p=2, B=10, F_l=32, filter='chebyshev5', pool='mpool1',
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0,
                regularization=0, batch_size=100, eval_frequency=200,
                dir_name='', V=None, task='classification', dropout=0.5, max_angle=360, mag_zoom=0.0, rand_noise=0.01, prob_augment=1):
        super().__init__()
        
        # Expand F, K and p according to the number of blocks B
        #F = np.tile(F, B);
        p = np.tile(p, B);
        K = np.tile(K, B);
        
        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
        
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        self.L_post = L[j]
        L = self.L
        
        
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        if V:
            Fin=V
        else:
            Fin=1
                
        
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        print(' ')
        print('Pre-convolution layer:')
        print('  biases: L_0 * F_0_ = {0} * {1} = {2}'.format(M_0, F[0], M_0*F[0]))
        print('  weights: V * F_0 * K_0 = {0} * {1} * {2} = {3}'.format(Fin, F[0], K[0], Fin*F[0]*K[0]))
        print(' ')
        print('ResNet blocks:')
        for i in range(Ngconv):
            print(' Block {0}'.format(i))
            print('    batch normalization: 2 * L_{0} = 2 * {1} = {2}'.format(i, L[i].shape[0], 2*L[i].shape[0]))
            print('    weights: F_{0} * F_{0} * K_{0} = {1} * {1} * {2} = {3}'.format(i,F[i], K[i], F[i]*F[i]*K[i]))
            
        print(' ')
        print('Post-convolution layer:')
        print('  batch normalization: 2 * L_{0} = 2 * {1} = {2}'.format(i, L[i].shape[0], 2*L[i].shape[0]))
        print(' ')
        print('Fully connected layer:')
        print('  weights: K_{0} * M = {1} * {2} = {3}'.format(i, K[i], M[0], K[i]*M[0]))
        
        
        # Store attributes and bind operations.
        ### Added by Pierre: V at the end of the line
        self.L, self.F, self.K, self.p, self.M, self.V = L, F, K, p, M, V
        ###
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization = regularization
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.pool = getattr(self, pool)
        self.task = task
        self.n_block = B
        self.F_post = F_l
        self.K_post = K[0]
        #self.K_post = 3
        self.dropout = dropout
        self.max_angle = max_angle
        self.mag_zoom = mag_zoom
        self.rand_noise = rand_noise
        self.prob_augment = prob_augment
        
        self.max_angle_radian = self.max_angle * 0.017453292519943295
        
        print('L_post = {0} ; F_post = {1} ; K_post = {2}'.format(self.L_post.shape[0], self.F_post, self.K_post))
        
        # a couple of things to store the best model
        if self.task == 'classification':
            self.best_accuracy = float('-inf')
        else:
            self.best_accuracy = float('inf')
        
        # Build the computational graph.
        self.build_graph(M_0)

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape() # batch_size x number_nodes x number_features
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        lmax = 1.02*scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]
        L = graph.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            #with tf.device('/cpu:0'):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        #for k in range(1, K):
        #    x1 = tf.sparse_tensor_dense_matmul(L, x0)  # M x Fin*N
        #    x = concat(x, x1)
        #    x0 = x1
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=True)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

#     def _inference(self, x, is_training):
#         # Graph convolutional layers.
#         ### Added by Pierre
#         if len(x.get_shape()) < 3:
#             x = tf.expand_dims(x, 2)  # N x M x F=1
#         ###
        
#         #x = self.batch_normalization_layer(x, is_training)        
#         for i in range(len(self.p)):
#             N, M, F = x.get_shape()
#             with tf.variable_scope('conv{}'.format(i+1)):
#                 #x = self.batch_normalization_layer(x, is_training)
#                 if not F == self.F[i]:
#                     with tf.variable_scope('linear_mapping'):
#                         x  = self.filter(x, self.L[i], self.F[i], self.K_post)
#                 for jj in range(1):
#                 #for jj in range(2):
#                     with tf.variable_scope('res_block{}'.format(jj+1)):
#                         with tf.variable_scope('Filter1'):
#                             x1 = self.batch_normalization_layer(x, is_training)
#                             x1 = tf.nn.relu(x1)
#                             x1 = self.filter(x1, self.L[i], self.F[i], self.K[i])
                            
#                         with tf.variable_scope('Filter2'):
#                             x1 = self.batch_normalization_layer(x1, is_training)
#                             x1 = tf.nn.relu(x1)
#                             x1 = self.filter(x1, self.L[i], self.F[i], self.K[i])
#                             x  = x + x1
                            
#                 # Pooling
#                 x  = self.pool(x, self.p[i])
                    
                    
#         N, M, F = x.get_shape()
#         with tf.variable_scope('post_block'):
#             #x = self.batch_normalization_layer(x, is_training)
#             if not F == self.F_post:
#                 with tf.variable_scope('linear_mapping'):
#                     x  = self.filter(x, self.L_post, self.F_post, self.K_post)
            
#             #x = tf.cond(tf.equal(is_training, 1), lambda: tf.nn.dropout(x, 0.5), lambda: tf.identity(x))
#             for jj in range(1):
#                     with tf.variable_scope('res_block{}'.format(jj+1)):
#                         with tf.variable_scope('Filter1'):
#                             x1  = self.batch_normalization_layer(x, is_training)
#                             x1 = tf.nn.relu(x1)
#                             x1 = self.filter(x1, self.L_post, self.F_post, self.K_post)
                            
#                         with tf.variable_scope('Filter2'):
#                             x1 = self.batch_normalization_layer(x1, is_training)
#                             x1 = tf.nn.relu(x1)
#                             x1 = self.filter(x1, self.L_post, self.F_post, self.K_post)
#                             x = x + x1
                            
#             # Add 1x1 convolution
#             #with tf.variable_scope('Convolution_1x1'):
#             #    #x = self.batch_normalization_layer(x, is_training)
#             #    x = self.filter(x, self.L_post, 64, 1)
                            
#             with tf.variable_scope('global_pooling'):
#                 #x = tf.nn.relu(x)
#                 #gap = tf.reduce_mean(x, 1) # Global pool -> N_subj (or batch_size) x F_l
                
#                 #### Added by Emanuel ####
#                 N, M, F = x.get_shape()
#                 gap = tf.reshape(x, [int(N), int(M*F)]) # NxM
#                 ##########################
                
#         # Logits linear layer, i.e. softmax without normalization.
#         N, Min = gap.get_shape()
#         with tf.variable_scope('fc_layer'):
#             #gap  = tf.cond(tf.equal(is_training, 1), lambda: tf.nn.dropout(gap, 0.5), lambda: tf.identity(gap))
#             W = self._weight_variable([int(Min), self.M[-1]], regularization=True)
#             x_fc = tf.matmul(gap, W)
            
# #             b = self._bias_variable([int(x_fc.shape[-2]),1], regularization=False)
            
                    
#         return x_fc
    def _inference(self, x, do_cam, is_training):
        # Graph convolutional layers.
        ### Added by Pierre
        if len(x.get_shape()) < 3:
            x = tf.expand_dims(x, 2)  # N x M x F=1
        ###
        
        #x = self.batch_normalization_layer(x, is_training)        
        for i in range(len(self.p)):
            N, M, F = x.get_shape()
            with tf.variable_scope('conv{}'.format(i+1)):
                #x = self.batch_normalization_layer(x, is_training)
                if not F == self.F[i]:
                    with tf.variable_scope('linear_mapping'):
                        x  = self.filter(x, self.L[i], self.F[i], self.K_post)
                for jj in range(1):
                #for jj in range(2):
                    with tf.variable_scope('res_block{}'.format(jj+1)):
                        with tf.variable_scope('Filter1'):
                            x1 = self.batch_normalization_layer(x, is_training)
                            x1 = tf.nn.relu(x1)
                            x1 = self.filter(x1, self.L[i], self.F[i], self.K[i])
                            
                        with tf.variable_scope('Filter2'):
                            x1 = self.batch_normalization_layer(x1, is_training)
                            x1 = tf.nn.relu(x1)
                            x1 = self.filter(x1, self.L[i], self.F[i], self.K[i])
                            x  = x + x1
                            
                # Pooling
                x  = self.pool(x, self.p[i])
                    
                    
        N, M, F = x.get_shape()
        with tf.variable_scope('post_block'):
            #x = self.batch_normalization_layer(x, is_training)
            if not F == self.F_post:
                with tf.variable_scope('linear_mapping'):
                    x  = self.filter(x, self.L_post, self.F_post, self.K_post)
            
            #x = tf.cond(tf.equal(is_training, 1), lambda: tf.nn.dropout(x, 0.5), lambda: tf.identity(x))
            for jj in range(1):
                    with tf.variable_scope('res_block{}'.format(jj+1)):
                        with tf.variable_scope('Filter1'):
                            x1  = self.batch_normalization_layer(x, is_training)
                            x1 = tf.nn.relu(x1)
                            x1 = self.filter(x1, self.L_post, self.F_post, self.K_post)
                            
                        with tf.variable_scope('Filter2'):
                            x1 = self.batch_normalization_layer(x1, is_training)
                            x1 = tf.nn.relu(x1)
                            x1 = self.filter(x1, self.L_post, self.F_post, self.K_post)
                            x = x + x1
                            
            # Add 1x1 convolution
            #with tf.variable_scope('Convolution_1x1'):
            #    #x = self.batch_normalization_layer(x, is_training)
            #    x = self.filter(x, self.L_post, 64, 1)
                            
            with tf.variable_scope('global_pooling'):
                # x = tf.nn.relu(x)
                # gap = tf.reduce_mean(x, 1) # Global pool -> N_subj (or batch_size) x F_l
                
                #### Added by Emanuel ####
                N, M, F = x.get_shape()
                gap = tf.reshape(x, [int(N), int(M*F)]) # NxM
                ##########################
                
        # Logits linear layer, i.e. softmax without normalization.
        N, Min = gap.get_shape()
        with tf.variable_scope('fc_layer'):
            # gap  = tf.cond(tf.equal(is_training, 1), lambda: tf.nn.dropout(gap, 0.8), lambda: tf.identity(gap))                #  change
            W = self._weight_variable([int(Min), self.M[-1]], regularization=True)
            x_fc = tf.matmul(gap, W)
            S,c = x_fc.get_shape()

            # get gradients with respect to each feature map, PER CLASS
            grad = [tf.gradients(x_fc[:,i], [x])[0] for i in range(c)]
            grad = [tf.expand_dims(g, axis = -1) for g in grad]
            grad = tf.concat(grad,axis = -1)
            ''' Added by Emanuel '''
            
            # GAP over the nodes dimension as you would over pixels in an image
            alpha_k = tf.reduce_mean(grad, axis = 1)
            
            lin_comb = tf.einsum('SKC,SNK->SNC', alpha_k, x)
            gradCAM = tf.nn.relu(lin_comb)


            # #### Added by Yunan
            # W_f = tf.reshape(W, [int(M), int(F),int(self.M[-1])])
            # cam = tf.einsum('ijk,jkp->ijp', x, W_f)
            # print(cam.shape)
            
#             b = self._bias_variable([int(x_fc.shape[-2]),1], regularization=False)
            
                    
        return x_fc, gradCAM   
    
    def batch_normalization_layer(self, x, is_training):
        '''
        Helper function to do batch normalziation
        '''
        is_training = tf.equal(is_training, 1)
        return tf.layers.batch_normalization(x, axis=-1, training=is_training, scale=False)
