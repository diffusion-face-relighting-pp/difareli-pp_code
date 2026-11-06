import glob
import os
import re
import sys
import torch as th

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.base_config import parse_args
from guided_diffusion.script_util import create_img_and_diffusion


def _resolve_device(device=None):
    if device is not None:
        return device
    if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0:
        return "cuda"
    return "cpu"


def _resolve_cfg_path(cfg_name):
    """Resolve YAML config file from a direct path or the repository's `config/` directory."""
    if os.path.isfile(cfg_name):
        return cfg_name

    config_root = os.path.join(REPO_ROOT, "config")
    if os.path.isdir(config_root):
        matches = [
            p
            for p in glob.glob(f"{config_root}/**", recursive=True)
            if os.path.isfile(p) and os.path.basename(p) == cfg_name
        ]
        if matches:
            return matches[0]

    raise FileNotFoundError(f"[#] Config '{cfg_name}' not found under {config_root}")


def _resolve_log_dir(log_dir):
    """Validate and return the checkpoint log directory."""
    if os.path.exists(log_dir):
        return log_dir
    raise FileNotFoundError(f"[#] Please check the model path at {log_dir}")


class CkptLoader:
    def __init__(self, log_dir, cfg_name, device=None) -> None:
        self.log_dir = log_dir
        self.cfg_name = cfg_name
        self.model_path = self.get_model_path()
        self.cfg = self.get_cfg()
        self.name = self.cfg.img_model.name
        self.device = _resolve_device(device)

    def get_cfg(self):
        cfg_file = _resolve_cfg_path(self.cfg_name)
        print("[#] Config Path : ", [cfg_file])
        return parse_args(ipynb={"mode": True, "cfg": cfg_file})

    def get_model_path(self):
        return _resolve_log_dir(self.log_dir)

    def available_model(self):
        avail_m = glob.glob(f"{self.model_path}/*.pt")
        filtered_m = []
        for m in avail_m:
            r = re.search(r"(_(\d+).pt)", m)
            if r:
                filtered_m.append(list(r.groups())[0])
        print("[#] Available ckpt : ", sorted(filtered_m))

    def load_model(self, ckpt_selector, step):
        if ckpt_selector == "ema":
            ckpt = f"ema_0.9999_{step}"
        elif ckpt_selector == "model":
            ckpt = f"model{step}"
        else:
            raise NotImplementedError(f"Unsupported ckpt_selector: {ckpt_selector}")

        self.available_model()
        model_dict, diffusion = create_img_and_diffusion(self.cfg)
        model_dict = {k: v for k, v in model_dict.items() if v is not None}

        for k, v in list(model_dict.items()):
            if isinstance(v, tuple):
                assert len(v) == 1
                model_dict[k] = v[0]

        for m_name in model_dict.keys():
            ckpt_path = f"{self.model_path}/{m_name}_{ckpt}.pt"
            print(f"[#] Loading...{ckpt_path}")
            model_dict[m_name].load_state_dict(th.load(ckpt_path, map_location="cpu"))
            model_dict[m_name].to(self.device)
            model_dict[m_name].eval()


        return model_dict, diffusion


class SimpleCkptLoader(CkptLoader):
    def __init__(self, model_path, cfg_path, device=None) -> None:
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.cfg_name = cfg_path
        self.cfg = self.get_cfg()
        self.name = self.cfg.img_model.name
        self.device = _resolve_device(device)
