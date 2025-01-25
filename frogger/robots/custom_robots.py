
from dataclasses import dataclass

from frogger import ROOT
from frogger.robots.robot_core import RobotModel, RobotModelConfig
from frogger.robots.robots import AlgrModel, AlgrModelConfig

class LeapModel(RobotModel):
    """The Allegro model."""

    def __init__(self, cfg: "LeapModelConfig") -> None:
        """Initialize the Allegro model."""
        self.cfg = cfg
        self.hand = cfg.hand
        super().__init__(cfg)

@dataclass(kw_only=True)
class LeapModelConfig(RobotModelConfig):
    """Configuration of the Algr robot model.

    Attributes
    ----------
    hand : str, default="lh"
        The hand to use. Can be "rh" or "lh".
    """

    hand: str = "lh"

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.hand in ["lh", "rh"]
        self.model_path = f"leap/robot.urdf"
        self.model_class = LeapModel
        if self.name is None:
            self.name = f"leap_{self.hand}"
        super().__post_init__()