import torch
import os


class SpeakerEncoderTrainer:
    """
    The class-manager used to train the Speaker Encoder model
    """

    def __init__(self, config, model, train_dataloader, test_loader, optimizer):
        """
        :param config: Speaker encoder yacs configurations
        :param model: Speaker encoder model
        :param train_dataloader: Train dataloader
        :param test_loader: Test dataloader
        :param optimizer: Optimizer used during training
        """

        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = config.DEVICE
        if self.config.TRAIN.CHECKPOINT_NAME is not None:
            self.checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR_PATH,
                                                self.config.TRAIN.CHECKPOINT_NAME)

        self.speakers_per_batch = config.TRAIN.SPEAKERS_PER_BATCH
        self.utterances_per_speaker = config.TRAIN.UTTERANCES_PER_SPEAKER
        self.learning_rate_init = self.config.TRAIN.LEARNING_RATE_INIT
        self.step = 1
        self.save_n_steps = self.config.TRAIN.SAVE_N_STEPS
        self.out_dir = self.config.TRAIN.OUT_DIR

        self.run_id = self.config.TRAIN.RUN_ID

    def init_training_session(self):
        """
        Inits the training procedure: creates necessary directories,
        load checkpoints and optimizer state if defined in configs

        :return: None
        """

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        if not os.path.exists(os.path.join(self.out_dir, "checkpoints")):
            os.mkdir(os.path.join(self.out_dir, "checkpoints"))

        if self.config.TRAIN.CHECKPOINT_NAME is not None:
            if self.checkpoint_path.exists():
                print("Found existing model \"%s\", loading it and resuming training." % self.run_id)
                checkpoint = torch.load(self.checkpoint_path)
                self.step = checkpoint["step"]
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.optimizer.param_groups[0]["lr"] = self.learning_rate_init
            else:
                print("No model \"%s\" found, starting training from scratch." % self.run_id)
        else:
            print("Starting the training from scratch.")

    def compute_loss(self, embeds):
        embeds_loss = embeds.view((self.speakers_per_batch, self.utterances_per_speaker, -1)).to(self.device)
        loss, eer = self.model.loss(embeds_loss)
        return loss, eer

    def train_one_step(self, speaker_batch):
        """
        One training step

        :param speaker_batch: Batch of input data
        :return: Values of loss at this step
        """

        self.model.train()
        inputs = torch.from_numpy(speaker_batch.data).to(self.device)
        embeds = self.model(inputs)

        loss, eer = self.compute_loss(embeds)

        # Backward pass
        self.model.zero_grad()
        loss.backward()
        self.model.do_gradient_ops()
        self.optimizer.step()
        return loss.item()

    def train(self, number_steps, each_n_print_steps=100):
        """
        Main training method

        :param number_steps: Number of training steps
        :param each_n_print_steps: Optional. Print the loss value each each_n_print_steps steps
        :return: None
        """

        for n_step, speaker_batch in enumerate(self.train_dataloader):
            loss_val = self.train_one_step(speaker_batch)
            if self.step % each_n_print_steps == 1:
                print(f'Step {self.step}. Train loss value: {loss_val}')
            self.step += 1
            if self.save_n_steps != 0 and n_step % self.save_n_steps == 0:
                print("Saving the model (step %d)" % self.step)
                torch.save({
                    "step": self.step,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, os.path.join(self.out_dir, "checkpoints", self.run_id + "_STEP_" + str(self.step) + ".pt"))

            if number_steps == n_step:
                print(f"Stopping Training Session at step #{number_steps}")
                break
        return None
