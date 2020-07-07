# build CNN model
import os
import sys
import logging
import logging.handlers as handlers
import json
import numpy as np

# open local settings
with open('./settings.json') as local_json_file:
    local_submodule_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])

# class definitions


class quality_factor:

    def detect(self, local_img):
        try:
            # analyze
            local_method = ''
        except Exception as e:
            print('error in quality_factor submodule')
            print(e)
            logger.error(str(e), exc_info=True)
        return local_method
