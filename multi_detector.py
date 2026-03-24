import sys
from pathlib import Path


if len(sys.argv) != 3:
    print('Usage: python multi_detector.py <category (wood/tile/...)> <image dir path>')
    sys.exit(1)

print('Importing libraries... It may take some time')
from anomalib.data import PredictDataset
from anomalib.models import Stfpm
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor

# Suppress all warnings and logging (for production)
import logging
import warnings
log = logging.getLogger("lightning_fabric")
log.setLevel('ERROR')
warnings.filterwarnings('ignore')

# Predictions
category = sys.argv[1]

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "models" / f"stfpm_{category}.ckpt"

engine = Engine()

dataset = PredictDataset(
    path=sys.argv[2].strip(),
    image_size=(256, 256),
)

# Custom threshold
post_processor = PostProcessor(
    image_sensitivity=0.3,
    pixel_sensitivity=0.3,
)
model = Stfpm(post_processor=post_processor)

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=CKPT_PATH,
)

for prediction in predictions:
    image_path = prediction.image_path
    anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
    pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
    pred_score = prediction.pred_score
    if bool(pred_label[0]):
        print(f'{image_path[0]} is anomalous with {round(100*float(pred_score[0]),2)}% certainty')
    else:
        print(f'{image_path[0]} is normal')


