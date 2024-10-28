import logging

def initialize_logging():
    logging.basicConfig(
        filename='logs/app.log', 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized.")