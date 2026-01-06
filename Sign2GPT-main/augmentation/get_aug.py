import importlib

def get_aug(aug_name, aug_params=None):
    if aug_name is not None:
        # Convert frozen ConfigDict to normal dict
        if aug_params is None:
            aug_params = {}
        else:
            aug_params = dict(aug_params)  # make mutable copy

        # Albumentations RandomResizedCrop in newer versions requires 'size'
        if "size" not in aug_params:
            aug_params["size"] = (224, 224)  # replace height + width

        mod = importlib.import_module(aug_name, package=None)
        return mod.Transformation(**aug_params)
    else:
        return None
