from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

class Trainer:
    def __init__(self, generator, discriminator, feature_extractor, lr_train_dir, hr_train_dir, lr_valid_dir, hr_valid_dir, learning_rate, loss_weights, losses, dataname, log_dirs, weights_generator, weights_discriminator, n_validation, flatness, fallback_save_every_n_epochs, adam_optimizer, metrics):
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.lr_train_dir = lr_train_dir
        self.hr_train_dir = hr_train_dir
        self.lr_valid_dir = lr_valid_dir
        self.hr_valid_dir = hr_valid_dir
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.losses = losses
        self.dataname = dataname
        self.log_dirs = log_dirs
        self.weights_generator = weights_generator
        self.weights_discriminator = weights_discriminator
        self.n_validation = n_validation
        self.flatness = flatness
        self.fallback_save_every_n_epochs = fallback_save_every_n_epochs
        self.adam_optimizer = adam_optimizer
        self.metrics = metrics
        self.callbacks = []

    def train(self, epochs, steps_per_epoch, batch_size, monitored_metrics):
        early_stop = EarlyStopping(monitor=monitored_metrics, patience=self.flatness, mode='min')
        tensor_board = TensorBoard(log_dir=self.log_dirs['tensor_board'])
        model_checkpoint1 = ModelCheckpoint(filepath=self.weights_generator, monitor=monitored_metrics, mode='min',
                                            save_best_only=True, save_weights_only=True, verbose=1)
        self.callbacks.append(early_stop)
        self.callbacks.append(tensor_board)
        self.callbacks.append(model_checkpoint1)
        if self.discriminator:
            model_checkpoint2 = ModelCheckpoint(filepath=self.weights_discriminator, monitor=monitored_metrics, mode='min',
                                                 save_best_only=True, save_weights_only=True, verbose=1)
            self.callbacks.append(model_checkpoint2)
        self.generator.compile(loss=self.losses, loss_weights=self.loss_weights, optimizer=self.adam_optimizer,
                               metrics=self.metrics)
        self.generator.fit_generator(self.prepare_data('train'), steps_per_epoch=steps_per_epoch, epochs=epochs,
                                      callbacks=self.callbacks, validation_data=self.prepare_data('valid'),
                                      validation_steps=self.n_validation, workers=2, use_multiprocessing=True)

    def prepare_data(self, type_data='train'):
        lr_image_list, hr_image_list = [], []

        input_file = os.path.join(self.lr_train_dir if type_data == 'train' else self.lr_valid_dir, self.dataname + '/*')
        output_file = os.path.join(self.hr_train_dir if type_data == 'train' else self.hr_valid_dir, self.dataname + '/*')

        input_list = sorted(glob(input_file))
        output_list = sorted(glob(output_file))

        for i, j in zip(input_list, output_list):
            assert i.split('/')[-1].split('.')[:-1] == j.split('/')[-1].split('.')[:-1]
            lr_image_list.append(imageio.imread(i))
            hr_image_list.append(imageio.imread(j))

        lr_image_array = np.array(lr_image_list)
        hr_image_array = np.array(hr_image_list)

        return [lr_image_array, hr_image_array]