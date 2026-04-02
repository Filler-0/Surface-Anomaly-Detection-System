import sys
from pathlib import Path


if len(sys.argv) != 3:
    print('Usage: python multi_detector.py <category (wood/tile/...)> <image dir path>')
    sys.exit(1)

print('Importing libraries... It may take some time')
from anomalib.data import PredictDataset
from anomalib.models import Stfpm
from anomalib.engine import Engine
# from anomalib.post_processing import PostProcessor

# Suppress all warnings and logging (for production)
import logging
import warnings
log = logging.getLogger("lightning_fabric")
log.setLevel('ERROR')
warnings.filterwarnings('ignore')

# Predictions
category = sys.argv[1]

BASE_DIR = Path(__file__).resolve().parent
NEW_CKPT  = BASE_DIR / "future_integration" / "models" / "new_stfpm" / f"stfpm_{category}.ckpt"
OLD_CKPT  = BASE_DIR / "models" / f"stfpm_{category}.ckpt"
CKPT_PATH = OLD_CKPT if OLD_CKPT.exists() else NEW_CKPT

engine = Engine()

dataset = PredictDataset(
    path=sys.argv[2].strip(),
    image_size=(256, 256),
)

model = Stfpm()

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=CKPT_PATH,
)

for prediction in predictions:
    image_path = prediction.image_path
    pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
    text_label = 'anomalous' if bool(pred_label[0]) else 'normal'
    pred_score = 100*float(prediction.pred_score[0])
    certainity_score = abs(50 - pred_score)*2
    additional_info = f'with {round(pred_score,2)}% certainty'*pred_label
    print(f'{image_path[0]} is {text_label}', additional_info)


