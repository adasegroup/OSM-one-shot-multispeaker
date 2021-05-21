import torch
import os


class SpeakerEncoderTrainer:
    def __init__(self, config, model, train_dataloader, test_loader, optimizer):
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

        self.out_checkpoints_dir = self.config.TRAIN.OUT_DIR
        self.run_id = self.config.TRAIN.RUN_ID

    def init_training_session(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        if not os.path.exists(os.path.join(self.out_dir, "checkpoints")):
            os.mkdir(os.path.join(self.out_dir, "checkpoints"))

        if self.module_configs.TRAIN.CHECKPOINT_NAME is not None:
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
        self.model.train()
        inputs = torch.from_numpy(speaker_batch.data).to(self.device)
        embeds = self.model(inputs)

        loss, eer = self.compute_loss(embeds)

        # Backward pass
        self.model.zero_grad()
        loss.backward()
        self.model.do_gradient_ops()
        self.optimizer.step()

    def train(self, number_steps):
        for n_step, speaker_batch in enumerate(self.train_dataloader):
            self.train_one_step(speaker_batch)
            self.step += 1
            if self.save_n_steps != 0 and n_step % self.save_n_steps == 0:
                print("Saving the model (step %d)" % self.step)
                torch.save({
                    "step": self.step,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, os.path.join(self.out_dir, "checkpoints", self.run_id + "_STEP_" + str(self.step) + ".pt"))

            if number_steps == n_step:
                print("Stopping Training Session")
                break




