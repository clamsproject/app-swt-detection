import yaml

import app
import modeling.backbones as backbones

train_config_fname = yaml.safe_load(open(app.default_config_fname))['model_config_file']
backbone_name = yaml.safe_load(open(train_config_fname))['img_enc_name']
backbones.model_map[backbone_name]()
