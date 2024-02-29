import yaml

import app
import modeling.backbones as backbones

for config_fname in app.default_model_storage.glob('*.yml'):
    backbones.model_map[yaml.safe_load(open(config_fname))['img_enc_name']]()
