import importlib

def get_aug(aug_name, aug_params={}):
    if aug_name is not None:
        # Ensure required parameters exist
        if "height" not in aug_params:
            aug_params["height"] = 224
        if "width" not in aug_params:
            aug_params["width"] = 224

        mod = importlib.import_module(aug_name, package=None)
        return mod.Transformation(**aug_params)
    else:
        return None
