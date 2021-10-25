import logging

"""
  Logging is the process to capture the flow of the code. When a function starts to exectute
    all the passed arguments values can be logged, the time the function starts or its
    return values.
      - allows to differentiate between prod environment and dev environment
      - provides module names where the log comes from
      - control to differentiate logs on the basis of severity
"""

def generate_logger(name, file_name):
  log_format = f"[%(levelname)s] - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"

  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)

  file_handler = logging.FileHandler(file_name)
  file_handler.setLevel(logging.INFO)

  formatter = logging.Formatter(log_format)
  file_handler.setFormatter(formatter)

  logger.addHandler(file_handler)
  return logger