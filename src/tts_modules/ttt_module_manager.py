import torch


class TTSModuleManager:
    def __init__(self, configs, model=None, test_dataloader=None, train_dataloader=None):
        self.configs = configs
        self.model_config = None
        self.__load_local_configs()
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.optimizer = None
        if self.model is None:
            self.__init_baseline_model()

    def load_model(self, url=None, verbose=True):
        if url is not None:
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location=self.device, progress=verbose)
        else:
            if "checkpoint_path" in self.model_config.keys():
                checkpoint = torch.load(self.model_config["checkpoint_path"], map_location=self.device)
            else:
                print("Please, provide URL to download weights or local path to load checkpoint")
                return None
        model_state_dict = checkpoint["model_state"] if "model_state" in checkpoint.keys() else checkpoint
        self.model.load_state_dict(model_state_dict)
        if "optimizer_state" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.model.eval()
        self.model = self.model.to(self.device)
        return None

    def save_model(self, path=None, epoch=None):
        state = {
            "model_state": self.model.state_dict()
        }
        if self.optimizer is not None:
            state["optimizer_state"] = self.optimizer.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        torch.save(state, path)
        return None

    def __load_local_configs(self):
        """
            Load all necessary configs
        """
        raise NotImplementedError

    def __init_baseline_model(self):
        """
            Initialize baseline model
        """
        raise NotImplementedError
