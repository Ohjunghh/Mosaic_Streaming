import torch
import logging


logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available, using CPU.")
