import argparse
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from utils import logger
from utils.import_utils import import_modules_from_folder

RegistryItem = TypeVar("RegistryItem", bound=Callable)


class Registry:
    def __init__(
        self,
        registry_name: str,
        base_class: Optional[type] = None,
        separator: Optional[str] = ":",
        lazy_load_dirs: Optional[List[str]] = None,
        internal_dirs: Sequence[str] = (),
    ) -> None:
        self.registry_name = registry_name
        self.base_class = base_class
        self.registry = {}
        self.arguments_accessed = False
        self.separator = separator
        self._modules_loaded = False
        self._lazy_load_dirs = lazy_load_dirs
        self.internal_dirs = internal_dirs
        if self._lazy_load_dirs is None:
            self._lazy_load_dirs = []

    def _load_all(self) -> None:
        if not self._modules_loaded:
            self._modules_loaded = True
            for dir_name in self._lazy_load_dirs:
                import_modules_from_folder(dir_name, extra_roots=self.internal_dirs)

    def items(self) -> List[Tuple[str, RegistryItem]]:
        self._load_all()
        return self.registry.items()

    def keys(self) -> List[str]:
        self._load_all()
        return self.registry.keys()

    def __iter__(self) -> Iterable[str]:
        self._load_all()
        return iter(self.registry)

    def __getitem__(self, key: Union[Tuple[str, str], str]) -> RegistryItem:
        self._load_all()

        type_ = None
        if isinstance(key, Tuple) and len(key) == 2:
            key, type_ = key

        assert isinstance(
            key, str
        ), f"Key should be an instance of string. Got {type(key)}"
        name, params = self.parse_key(key)
        if type_:
            name = f"{type_}{self.separator}{name}"

        if name not in self.registry:
            registry_keys = list(self.registry.keys())
            temp_str = (
                f"\n{name} not yet supported in {self.registry_name} registry."
                f"\nSupported values are:"
            )
            for i, supp_val in enumerate(registry_keys):
                temp_str += f"\n\t {i}: {supp_val}"
            logger.error(temp_str + "\n")

        reg_item = self.registry[name]

        if params:
            reg_item = partial(reg_item, **params)
        return reg_item

    def __contains__(self, key: str) -> bool:
        self._load_all()
        name, _ = self.parse_key(key)
        return name in self.registry

    def register(self, name: str, type: str = "") -> Callable:
        if type:
            name = "{}{}{}".format(type, self.separator, name)

        if self.arguments_accessed:
            logger.error(
                f"Found item `{name}` being registered after all_item_arguments"
                f" was called for `{self.registry_name}` registry."
            )

        def register_with_name(item: RegistryItem) -> RegistryItem:
            if name in self.registry:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(self.registry_name, name)
                )
            if self.base_class and not issubclass(item, self.base_class):
                raise ValueError(
                    "{} class ({}: {}) must extend {}".format(
                        self.registry_name, name, item.__name__, self.base_class
                    )
                )

            self.registry[name] = item
            return item

        return register_with_name

    def all_arguments(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        self._load_all()
        self.arguments_accessed = True

        for _, item in self.items():
            parser = item.add_arguments(parser)

        return parser

    def parse_key(self, key: str) -> Tuple[str, Dict[str, str]]:
        name = key.split("(")[0]

        params = {}
        if "(" in key:
            params_str = key.split("(")[1].split(")")[0]

            try:
                params = dict(
                    [
                        [x.strip() for x in arg.split("=")]
                        for arg in params_str.split(",")
                    ]
                )
            except Exception as e:
                logger.error(
                    "Could not correctly parse key parameters `{}` for registry {}."
                    " Please make sure to key parameters have the format:"
                    " <key_name>(arg1=value1, arg2=value2, ...)".format(
                        key, self.registry_name
                    )
                )
                raise e

        return name, params
