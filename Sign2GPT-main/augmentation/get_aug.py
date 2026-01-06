import importlib

def get_aug(aug_name, aug_params=None):
    if aug_name is not None:
        # Make a mutable dict
        if aug_params is None:
            aug_params = {}
        else:
            aug_params = dict(aug_params)  # convert ConfigDict to normal dict

        # Ensure required parameters exist
        if "size" not in aug_params:  # Albumentations new version
            aug_params["size"] = (224, 224)

        mod = importlib.import_module(aug_name, package=None)
        return mod.Transformation(**aug_params)
    else:
        return None
