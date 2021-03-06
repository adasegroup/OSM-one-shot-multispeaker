import torch
import requests
import os


class AbstractTTSModuleManager:
    """
    An abstract class used as a basis for particular module managers

    Attributes
    -----------
    main_configs: yacs.config.CfgNode
        main configurations
    module_configs: yacs.config.CfgNode
        module configurations
    model: torch.nn.Module
        Pytorch NN model
    model_name: str
        Name of the model
    optimizer: nn.optim.Optimizer
        Pytorch optimizer
    test_dataloader: torch.utils.data.Dataloader
        Test dataloader
    train_dataloader: torch.utils.data.Dataloader
        Train dataloader

    Methods
    ------------
    load_model(...)
        Loads the weights of the model from the local checkpoint or download the checkpoint if url is given

    save_model(..)
        Saves the state of the model and optimizer

    _load_local_configs()
        Loads yacs configs for the module

    _init_baseline_model()
        Initializes baseline model

    _load_baseline_model()
        Load the checkpoint for the baseline model

    __load_from_dropbox()
        Downloads the checkpoint from the Dropbox link

    """

    def __init__(self,
                 main_configs,
                 model=None,
                 test_dataloader=None,
                 train_dataloader=None,
                 optimizer=None
                 ):
        self.main_configs = main_configs
        self.module_configs = None
        self.model_name = None
        self._load_local_configs()
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.optimizer = optimizer
        if self.model is None:
            self._init_baseline_model()

    def load_model(self, url=None, verbose=True):
        """
        Loads the weights of the model from the local checkpoint or download the checkpoint if url is given

        :param url: URL to the remote store
        :param verbose: Flag defines whether to print info or not
        :return: None
        """

        if url is not None:
            checkpoint = torch.hub.load_state_dict_from_url(url,
                                                            map_location=self.device,
                                                            progress=verbose
                                                            )
        else:
            if self.module_configs.MODEL.CHECKPOINT_DIR_PATH is not None:
                checkpoint = torch.load(self.module_configs.MODEL.CHECKPOINT_DIR_PATH,
                                        map_location=self.device
                                        )
            else:
                print("Please, provide URL to download weights or local path to load checkpoint")
                return None
        self.__load_routine(checkpoint)
        return None

    def save_model(self, path=None, epoch=None):
        """
        Saves the states of the model and optimizer and the current epoch number

        :param path: Path to the checkpoint file
        :param epoch: Epoch number
        :return: None
        """

        state = {
            "model_state": self.model.state_dict()
        }
        if self.optimizer is not None:
            state["optimizer_state"] = self.optimizer.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        torch.save(state, path)
        return None

    def __load_routine(self, checkpoint):
        model_state_dict = checkpoint["model_state"] if "model_state" in checkpoint.keys() else checkpoint
        self.model.load_state_dict(model_state_dict)
        if "optimizer_state" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.model.eval()
        self.model = self.model.to(self.device)
        return None

    def _load_baseline_model(self):
        checkpoint = self.__get_baseline_checkpoint()
        self.__load_routine(checkpoint)
        return None

    def __get_baseline_checkpoint(self):
        if not os.path.exists(self.module_configs.MODEL.CHECKPOINT_DIR_PATH):
            os.mkdir(self.module_configs.MODEL.CHECKPOINT_DIR_PATH)
        file_full_path = os.path.join(
            self.module_configs.MODEL.CHECKPOINT_DIR_PATH,
            self.model.get_download_url().split('/')[-1]
        )
        if not os.path.isfile(file_full_path):
            self.__load_from_dropbox(file_full_path)
        else:
            if self.module_configs.VERBOSE:
                print(f'Loading {self.model_name} checkpoint from {file_full_path}')
        checkpoint = torch.load(file_full_path, map_location=self.device)
        return checkpoint

    def __load_from_dropbox(self, file_full_path):
        if self.module_configs.VERBOSE:
            print(f'Downloading {self.model_name} checkpoint from Dropbox...')
        try:
            req = requests.get(self.model.get_download_url())
            with open(file_full_path, 'wb') as f:
                f.write(req.content)
        except requests.exceptions.RequestException as e:
            print(f'Baseline {self.model_name} checkpoint was not loaded from Dropbox!')
            print(f'Stacktrace: {e}')
        return None

    def _load_local_configs(self):
        """
        Load all necessary main_configs
        """
        raise NotImplementedError

    def _init_baseline_model(self):
        """
        Initialize baseline model
        """
        raise NotImplementedError
