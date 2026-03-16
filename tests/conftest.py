import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model": {
            "name": "Restore_RWKV",
            "channels": 3,
            "layers": 4,
            "hidden_dim": 128,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "epochs": 10,
            "device": "cpu",
        },
        "data": {
            "root_dir": "/tmp/test_data",
            "modalities": ["CT", "MRI", "PET"],
            "image_size": 256,
        },
        "paths": {
            "checkpoint_dir": "/tmp/checkpoints",
            "log_dir": "/tmp/logs",
            "output_dir": "/tmp/outputs",
        }
    }


@pytest.fixture
def sample_image_tensor() -> torch.Tensor:
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def batch_image_tensor() -> torch.Tensor:
    """Create a batch of image tensors for testing."""
    return torch.randn(4, 3, 256, 256)


@pytest.fixture
def sample_numpy_image() -> np.ndarray:
    """Create a sample numpy image array."""
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a sample PIL image."""
    array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    model.state_dict = MagicMock(return_value={"layer1.weight": torch.randn(10, 10)})
    model.load_state_dict = MagicMock()
    model.cuda = MagicMock(return_value=model)
    model.cpu = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.forward = MagicMock(return_value=torch.randn(4, 3, 256, 256))
    return model


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = MagicMock()
    dataloader.__iter__ = MagicMock(return_value=iter([
        (torch.randn(4, 3, 256, 256), torch.randn(4, 3, 256, 256))
        for _ in range(5)
    ]))
    dataloader.__len__ = MagicMock(return_value=5)
    return dataloader


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    optimizer.state_dict = MagicMock(return_value={"state": {}, "param_groups": []})
    optimizer.load_state_dict = MagicMock()
    return optimizer


@pytest.fixture
def mock_scheduler():
    """Create a mock learning rate scheduler."""
    scheduler = MagicMock()
    scheduler.step = MagicMock()
    scheduler.get_last_lr = MagicMock(return_value=[1e-4])
    return scheduler


@pytest.fixture
def sample_medical_data(temp_dir: Path) -> Dict[str, Path]:
    """Create sample medical imaging data files."""
    data_paths = {}
    modalities = ["CT", "MRI", "PET"]
    
    for modality in modalities:
        modality_dir = temp_dir / modality
        modality_dir.mkdir(exist_ok=True)
        
        for i in range(3):
            img_path = modality_dir / f"sample_{i:03d}.png"
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            )
            img.save(img_path)
            
        data_paths[modality] = modality_dir
    
    return data_paths


@pytest.fixture
def mock_loss_function():
    """Create a mock loss function."""
    loss_fn = MagicMock()
    loss_fn.forward = MagicMock(return_value=torch.tensor(0.5))
    loss_fn.__call__ = MagicMock(return_value=torch.tensor(0.5))
    return loss_fn


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_transform():
    """Create a mock transform function."""
    def transform(x):
        if isinstance(x, Image.Image):
            return torch.randn(3, 256, 256)
        return x
    return transform


@pytest.fixture
def sample_checkpoint(temp_dir: Path) -> Path:
    """Create a sample checkpoint file."""
    checkpoint_path = temp_dir / "checkpoint.pth"
    checkpoint = {
        "epoch": 10,
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "loss": 0.123,
        "metrics": {"psnr": 30.5, "ssim": 0.89}
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    buffer = io.StringIO()
    
    def get_output():
        return buffer.getvalue()
    
    monkeypatch.setattr(sys, "stdout", buffer)
    return get_output


@pytest.fixture
def mock_tqdm(mocker):
    """Mock tqdm progress bar."""
    return mocker.patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x)


@pytest.fixture
def isolated_filesystem(tmp_path):
    """Create an isolated filesystem for testing."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Provide sample evaluation metrics."""
    return {
        "psnr": 28.5,
        "ssim": 0.85,
        "mae": 0.015,
        "mse": 0.0025,
        "ncc": 0.92,
    }


@pytest.fixture
def mock_cuda_kernel(mocker):
    """Mock CUDA kernel operations."""
    mock = mocker.patch("torch.utils.cpp_extension.load")
    mock.return_value.forward = MagicMock(return_value=torch.randn(4, 128, 256, 256))
    mock.return_value.backward = MagicMock(return_value=(torch.randn(4, 128, 256, 256),))
    return mock