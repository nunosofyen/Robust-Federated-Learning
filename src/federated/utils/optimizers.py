from tensorflow.keras.optimizers.schedules import CosineDecay


class CustomCosineDecay(CosineDecay):
    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)

            global_step_recomp = tf.cast(step, dtype)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
            completed_fraction = global_step_recomp / decay_steps
            cosine_decayed = 0.5 * (1.0 + tf.cos(
                tf.constant(math.pi, dtype=dtype) * completed_fraction))

            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(initial_learning_rate, decayed)