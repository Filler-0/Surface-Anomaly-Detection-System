import sys


if len(sys.argv) != 2:
    print('Usage: python wood_detector.py <image dir path>')
    sys.exit(1)

print('Importing libraries... It may take some time')
from anomalib.data import PredictDataset
from anomalib.models import Stfpm
from anomalib.engine import Engine

# Suppress all warnings and logging (for production)
import logging
import warnings
log = logging.getLogger("lightning_fabric")
log.setLevel('ERROR')
warnings.filterwarnings('ignore')

# Predictions
engine = Engine()
model = Stfpm()

dataset = PredictDataset(
    path=sys.argv[1].strip(),
    image_size=(256, 256),
)

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path="stfpm.ckpt",
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


