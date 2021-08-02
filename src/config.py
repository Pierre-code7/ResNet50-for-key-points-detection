import torch

# constant paths
ROOT_PATH = '/home/pito/Documents/Sensor6ty/keypoints_model/input'
OUTPUT_PATH = '/content/drive/MyDrive/consultant/PointsRemarquablesDataset/input_key_points/input'

# learning parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split
TEST_SPLIT = 0.1

# show dataset keypoint plot
SHOW_DATASET_PLOT = True
