import logging
from dataclasses import dataclass


# === MOCK CLASSES ===
@dataclass
class ManagedObject:
    species: str = ""
    bounding_box_size: float = 0.0
    confidence: float = 1.0


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SPRAY_SPECIES = {"TAROF", "CHEAL", "EQUAR", "1MATG",
                 "GALAP", "SINAR", "1CRUF", "CIRAR", "POLCO"}
PRESERVE_SPECIES = {"VIOAR", "GERMO", "EPHHE", "LAMPU"}
CHECK_SIZE_SPECIES = {"FUMOF", "POLLA", "POLAV", "ATXPA", "VERPE"}
CONFIDENCE = 0.4


class MarcoStrategy:

    def __init__(self) -> None:
        self.spray_list = SPRAY_SPECIES
        self.preserve_list = PRESERVE_SPECIES
        self.check_size_list = CHECK_SIZE_SPECIES
        self.confidence = CONFIDENCE

    def decide(self, target: ManagedObject) -> bool:
        """
        Decide to spray or preserve based on the species, size, and confidence.
        Returns True for spray, False for preserve.
        """
        species = target.species
        size = target.bounding_box_size
        confidence = target.confidence

        # below threshold confidence, do not spray
        if confidence < self.confidence:
            return False

        if species in self.spray_list:
            return True
        elif species in self.preserve_list:
            return False
        elif species in self.check_size_list:
            # Check the size condition
            if size > 40:
                return True
            else:
                return False
        else:
            # For unknown species, check the size condition
            if size > 40:
                return True
            else:
                return False


if __name__ == "__main__":

    strategy = MarcoStrategy()

    target = ManagedObject(
        species="TAROF", bounding_box_size=50, confidence=0.4)
    decision = strategy.decide(target)
    logger.info(f"Decision to spray: {decision}")

    target = ManagedObject(
        species="VIOAR", bounding_box_size=30, confidence=0.2)
    decision = strategy.decide(target)
    logger.info(f"Decision to spray: {decision}")

    target = ManagedObject(
        species="FUMOF", bounding_box_size=30, confidence=0.9)
    decision = strategy.decide(target)
    logger.info(f"Decision to spray: {decision}")

    target = ManagedObject(
        species="FUMOF", bounding_box_size=50, confidence=0.5)
    decision = strategy.decide(target)
    logger.info(f"Decision to spray: {decision}")
