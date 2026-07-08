from pathlib import Path

import hydra
from omegaconf import DictConfig

from DL_Template.utils import get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)

    if not cfg.data.source_url:
        log.warning("cfg.data.source_url is not set; nothing to download.")
        return

    log.info("Downloading dataset from %s to %s", cfg.data.source_url, root)
    # TODO: implement actual download/extraction logic for your dataset.


if __name__ == "__main__":
    main()
