from pathlib import Path


def get_checkpoint_path(base_name, name):
    ckpt_path = "/kaggle/working/signLLM/Sign2GPT-main/checkpoints"
    return ckpt_path


def get_lmdb_path():
    lmdb_path = "/kaggle/input/phoenixweather2014t-3rd-attempt"
    return lmdb_path
