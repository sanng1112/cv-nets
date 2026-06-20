class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass


class LayerDefinitionError(Exception):
    """Raised when a layer class definition violates expected patterns."""
    pass


class ModelBuildError(Exception):
    """Raised when model construction from config fails."""
    pass
