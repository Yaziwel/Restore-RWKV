import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image


class TestInfrastructureValidation:
    """Validation tests to ensure the testing infrastructure is properly set up."""
    
    def test_python_version(self):
        """Test that Python version meets requirements."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_pytest_available(self):
        """Test that pytest is properly installed."""
        assert pytest.__version__, "pytest should be installed"
    
    def test_coverage_plugin(self):
        """Test that pytest-cov plugin is available."""
        import pytest_cov
        assert pytest_cov, "pytest-cov should be installed"
    
    def test_mock_plugin(self):
        """Test that pytest-mock plugin is available."""
        import pytest_mock
        assert pytest_mock, "pytest-mock should be installed"
    
    def test_torch_available(self):
        """Test that PyTorch is properly installed."""
        assert torch.__version__, "PyTorch should be installed"
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.shape == (3,)
    
    def test_numpy_available(self):
        """Test that NumPy is properly installed."""
        assert np.__version__, "NumPy should be installed"
        array = np.array([1, 2, 3])
        assert array.shape == (3,)
    
    def test_pillow_available(self):
        """Test that Pillow is properly installed."""
        assert Image, "Pillow should be installed"
        img = Image.new('RGB', (10, 10))
        assert img.size == (10, 10)
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that the unit test marker works."""
        assert True, "Unit test marker should work"
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that the integration test marker works."""
        assert True, "Integration test marker should work"
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that the slow test marker works."""
        assert True, "Slow test marker should work"
    
    def test_project_imports(self):
        """Test that project modules can be imported."""
        try:
            from tools import set_seeds, mkdir
            assert set_seeds is not None
            assert mkdir is not None
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
    
    def test_data_module_imports(self):
        """Test that data modules can be imported."""
        try:
            from data.common import transformData, dataIO
            assert transformData is not None
            assert dataIO is not None
        except ImportError as e:
            pytest.fail(f"Failed to import data modules: {e}")
    
    @pytest.mark.skip(reason="Model requires CUDA kernel compilation")
    def test_model_module_imports(self):
        """Test that model modules can be imported."""
        try:
            from model import Restore_RWKV
            assert Restore_RWKV is not None
        except ImportError as e:
            pytest.fail(f"Failed to import model modules: {e}")
    
    def test_evaluation_module_imports(self):
        """Test that evaluation modules can be imported."""
        try:
            from evaluation.evaluation_metric import compute_measure
            assert compute_measure is not None
        except ImportError as e:
            pytest.fail(f"Failed to import evaluation modules: {e}")
    
    def test_loss_module_imports(self):
        """Test that loss modules can be imported."""
        try:
            from loss import losses
            assert losses is not None
        except ImportError as e:
            pytest.fail(f"Failed to import loss modules: {e}")


class TestFixturesValidation:
    """Tests to validate that all fixtures work correctly."""
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture creates and cleans up properly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
    
    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture provides valid configuration."""
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
        assert "paths" in mock_config
        assert mock_config["model"]["name"] == "Restore_RWKV"
    
    def test_sample_image_tensor_fixture(self, sample_image_tensor):
        """Test that sample_image_tensor fixture creates valid tensor."""
        assert isinstance(sample_image_tensor, torch.Tensor)
        assert sample_image_tensor.shape == (1, 3, 256, 256)
    
    def test_batch_image_tensor_fixture(self, batch_image_tensor):
        """Test that batch_image_tensor fixture creates valid batch."""
        assert isinstance(batch_image_tensor, torch.Tensor)
        assert batch_image_tensor.shape == (4, 3, 256, 256)
    
    def test_sample_numpy_image_fixture(self, sample_numpy_image):
        """Test that sample_numpy_image fixture creates valid array."""
        assert isinstance(sample_numpy_image, np.ndarray)
        assert sample_numpy_image.shape == (256, 256, 3)
        assert sample_numpy_image.dtype == np.float32
    
    def test_sample_pil_image_fixture(self, sample_pil_image):
        """Test that sample_pil_image fixture creates valid PIL image."""
        assert isinstance(sample_pil_image, Image.Image)
        assert sample_pil_image.size == (256, 256)
    
    def test_mock_model_fixture(self, mock_model):
        """Test that mock_model fixture works correctly."""
        assert hasattr(mock_model, 'eval')
        assert hasattr(mock_model, 'train')
        assert hasattr(mock_model, 'parameters')
        result = mock_model.forward(torch.randn(4, 3, 256, 256))
        assert result.shape == (4, 3, 256, 256)
    
    def test_mock_dataloader_fixture(self, mock_dataloader):
        """Test that mock_dataloader fixture works correctly."""
        assert len(mock_dataloader) == 5
        for batch in mock_dataloader:
            assert len(batch) == 2
            break
    
    def test_mock_optimizer_fixture(self, mock_optimizer):
        """Test that mock_optimizer fixture works correctly."""
        assert hasattr(mock_optimizer, 'zero_grad')
        assert hasattr(mock_optimizer, 'step')
        mock_optimizer.zero_grad()
        mock_optimizer.step()
    
    def test_sample_medical_data_fixture(self, sample_medical_data):
        """Test that sample_medical_data fixture creates valid data."""
        assert "CT" in sample_medical_data
        assert "MRI" in sample_medical_data
        assert "PET" in sample_medical_data
        for modality, path in sample_medical_data.items():
            assert path.exists()
            assert path.is_dir()
            images = list(path.glob("*.png"))
            assert len(images) == 3
    
    def test_device_fixture(self, device):
        """Test that device fixture returns valid device."""
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
    
    def test_random_seed_fixture(self, random_seed):
        """Test that random_seed fixture sets seeds properly."""
        assert random_seed == 42
        tensor1 = torch.randn(10)
        torch.manual_seed(42)
        tensor2 = torch.randn(10)
        assert torch.allclose(tensor1, tensor2)
    
    def test_sample_checkpoint_fixture(self, sample_checkpoint):
        """Test that sample_checkpoint fixture creates valid checkpoint."""
        assert sample_checkpoint.exists()
        checkpoint = torch.load(sample_checkpoint, map_location='cpu')
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 10
    
    def test_sample_metrics_fixture(self, sample_metrics):
        """Test that sample_metrics fixture provides valid metrics."""
        assert "psnr" in sample_metrics
        assert "ssim" in sample_metrics
        assert sample_metrics["psnr"] == 28.5
        assert sample_metrics["ssim"] == 0.85


@pytest.mark.parametrize("test_input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_parametrized_example(test_input, expected):
    """Example of parametrized test to verify functionality."""
    assert test_input ** 2 == expected


def test_skip_example():
    """Example of a test that can be skipped."""
    if sys.platform == "win32":
        pytest.skip("Skipping on Windows")
    assert True


def test_xfail_example():
    """Example of a test expected to fail."""
    pytest.xfail("This test is expected to fail")
    assert False