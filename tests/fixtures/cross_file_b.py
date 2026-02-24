from cross_file_a import get_logger


def process(data: str) -> None:
    logger = get_logger()
    logger.info(f"Processing: {data}")
