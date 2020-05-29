import tensorflow as tf

class WarmupCosineDecayLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 max_lr: float,
                 min_lr: float,
                 warmup_steps: int,
                 decay_steps: int,
                 alpha: float = 0.):
        super().__init__()

        self.name = 'WarmupCosineDecayLRScheduler'
        self.alpha = alpha

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.last_step = 0

        self.warmup_steps = warmup_steps
        self.linear_increase = (self.max_lr - self.min_lr) / float(self.warmup_steps)
        self.decay_steps = decay_steps

    def run_decay(self):
        step = tf.cast(self.last_step - self.warmup_steps, tf.float32)

        cosine_decay = 0.5 * (1 + tf.cos(3.1415 * step / float(self.decay_steps)))
        decayed = (1. - self.alpha) * cosine_decay + self.alpha
        return self.max_lr * decayed

    @property
    def current_lr(self):
        return tf.cond(tf.less(self.last_step, self.warmup_steps),
                       lambda: tf.multiply(self.linear_increase, self.last_step),
                       lambda: self.run_decay())

    def __call__(self, epoch, step, new_metric):
        self.last_step = step
        return self.current_lr, False

    def get_config(self):
        config = {
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha
        }
        return config

class ResetLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 learning_rate: float,
                 min_learning_rate: float,
                 epochs_lr_update: int,
                 warmup_steps: int = 0,
                 reset_on_lr_update: bool = False,
                 initial_learning_rate_multiplier: float = 0.2):
        super().__init__()

        self.name = 'ResetLRScheduler'

        self.learning_rate = learning_rate
        self.initial_learning_rate_multiplier = initial_learning_rate_multiplier
        self.reset_on_lr_update = reset_on_lr_update

        self.warmup_steps = warmup_steps

        self.reset(0)

    def reset(self, best_metric):
        self.num_epochs_without_improvement = 0
        self.learning_rate_multiplier = self.initial_learning_rate_multiplier
        self.best_metric = best_metric

    def __call__(self, epoch, step, new_metric):
        if new_metric > self.best_metric:
            self.reset(new_metric)
        else:
            self.num_epochs_without_improvement += 1

        if steps < self.warmup_steps:
            return self.learning_rate, False

        if self.num_epochs_without_improvement >= self.epochs_lr_update:
            if self.learning_rate > self.min_learning_rate:
                old_lr = self.learning_rate

                self.learning_rate *= self.learning_rate_multiplier
                if self.learning_rate < self.min_learning_rate:
                    self.learning_rate = self.min_learning_rate

                if self.reset_on_lr_update:
                    want_reset = True

                logger.info('epoch: {}, global_step: {}, epochs without metric improvement: {}, metric: {:.5f}/{:.5f}, updating learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    epoch, step, self.num_epochs_without_improvement, new_metric, self.best_metric, old_lr, self.learning_rate, want_reset))

                self.num_epochs_without_improvement = 0
                if self.learning_rate_multiplier > 0.1:
                    self.learning_rate_multiplier /= 2

            else:
                self.learning_rate = self.initial_learning_rate
                want_reset = True

                logger.info('epoch: {}/{}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    epoch_var.numpy(), num_epochs_without_improvement, global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier

        return self.learning_rate, want_reset
