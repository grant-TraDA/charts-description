class Config():

    def __init__(self, model, X_train_path, X_val_path, default_path, target_size, batch_size, epoch,
                 steps_per_epoch, validation_steps, train_frac, color_mode, n_hidden_layers, num_classes):
        self.model = model
        self.X_train_path = X_train_path
        self.X_val_path = X_val_path
        self.default_path = default_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.train_frac = train_frac
        self.color_mode = color_mode
        self.n_hidden_layers = n_hidden_layers
        self.num_classes = num_classes
