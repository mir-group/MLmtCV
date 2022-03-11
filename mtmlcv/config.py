"""
Class to holde a bunch of hyperparameters associate with either training or a model.

The interface is inteneded to be as close to the wandb.config class as possible. But it does not have any locked
entries as in wandb.config

Examples:

    Initialization
    ```
    config = Config()
    config = Config(dict(a=1, b=2))
    ```

    add a new parameter

    ```
    config['key'] = default_value
    config.key = default_value
    ```

    set up typehint for a parameter
    ```
    config['_key_type'] = int
    config._key_type = int
    config.set_type(key, int)
    ```

    update with a dictionary
    ```
    config.update(dictionary={'a':3, 'b':4})
    ```

    If a parameter is updated, the updated value will be formatted back to the same type.

"""
from copy import deepcopy

from mtmlcv.savenload import save_file, load_file


class Config(object):
    def __init__(self, config: dict = None, allow_list: list = None):

        object.__setattr__(self, "_items", dict())
        object.__setattr__(self, "_item_types", dict())
        object.__setattr__(self, "_allow_list", list())
        object.__setattr__(self, "_allow_all", True)

        if config is not None:
            self.update(config)
        if allow_list is not None:
            self.set_allow_list(allow_list)

    def __repr__(self):
        return str(dict(self))

    __str__ = __repr__

    def keys(self):
        return self._items.keys()

    def _as_dict(self):
        return self._items

    def as_dict(self):
        return dict(self)

    def __getitem__(self, key):
        return self._items[key]

    def get_type(self, key):
        """Get Typehint from item_types dict or previous defined value

        If there's a value already stored in the list, get the type from the value.
        But this can be overwritten by the typehint that were explicitely
        stated

        Args:

            key: name of the variable
        """

        ori_val = self._items.get(key, None)
        typehint = type(ori_val) if ori_val is not None else None
        typehint = self._item_types.get(key, typehint)
        return typehint

    def set_type(self, key, typehint):
        """set typehint for a variable

        Args:

            key: name of the variable
            typehint: type of the variable
        """

        self._item_types[key] = typehint

    def set_allow_list(self, keys, default_value=None):
        object.__setattr__(self, "_allow_all", False)
        object.__setattr__(self, "_allow_list", deepcopy(keys))
        for key in keys:
            self.__setitem__(key, default_value)

    def __setitem__(self, key, val):

        # typehint
        if key.endswith("_type") and key.startswith("_"):

            k = key[1:-5]
            if (not self._allow_all) and key not in self._allow_list:
                return

            self._item_types[k] = val

        # normal value
        else:

            if (not self._allow_all) and key not in self._allow_list:
                return

            typehint = self.get_type(key)

            # try to format the variable
            try:
                val = typehint(val) if typehint is not None else val
            except:
                raise TypeError(
                    f"Wrong Type: Parameter {key} should be {typehint} type."
                    "But {type(val)} is given"
                )

            self._items[key] = deepcopy(val)

    def items(self):
        return self._items.items()

    __setattr__ = __setitem__

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __contains__(self, key):
        return key in self._items

    def update(self, dictionary: dict, allow_val_change=None):
        """Mock of wandb.config function

        Add a dictionary of parameters to the config
        The key of the parameter cannot be started as "_"

        Args:

            dictionary (dict): dictionary of parameters and their typehint to update
            allow_val_change (None): mock for wandb.config, not used.

        Returns:

        """

        # first log in all typehints or hidden variables
        for k, value in dictionary.items():
            if k.startswith("_"):
                self.__setitem__(k, value)

        # then log in the values
        for k, value in dictionary.items():
            if not k.startswith("_"):
                self.__setitem__(k, value)

    def get(self, *args):
        return self._items.get(*args)

    def persist(self):
        """mock wandb.config function"""
        pass

    def setdefaults(self, d):
        """mock wandb.config function"""
        pass

    def update_locked(self, d, user=None):
        """mock wandb.config function"""
        pass

    def save(self, filename, format: str = None):
        """ Print config to file. """

        supported_formats = {"yaml": ("yml", "yaml"), "json": "json"}
        return save_file(
            item=dict(self),
            supported_formats=supported_formats,
            filename=filename,
            enforced_format=format,
        )

    @staticmethod
    def from_file(filename, format: str = None):
        """ Load arguments from file """

        supported_formats = {"yaml": ("yml", "yaml"), "json": "json"}
        dictionary = load_file(
            supported_formats=supported_formats,
            filename=filename,
            enforced_format=format,
        )
        c = Config()
        c.update(dictionary)
        return c

    load = from_file
