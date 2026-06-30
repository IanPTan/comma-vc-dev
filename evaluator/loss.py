import torch
import math
from safetensors.torch import load_file
from evaluator.models import PoseNet, SegNet, DEFAULT_POSENET_WEIGHTS, DEFAULT_SEGNET_WEIGHTS

class EvaluatorManager:
    def __init__(self, device=None, posenet_weights=None, segnet_weights=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)
        
        self.posenet_weights = posenet_weights or DEFAULT_POSENET_WEIGHTS
        self.segnet_weights = segnet_weights or DEFAULT_SEGNET_WEIGHTS
        
        self._posenet = None
        self._segnet = None

    @property
    def posenet(self):
        if self._posenet is None:
            model = PoseNet().eval().to(self.device)
            sd = load_file(self.posenet_weights, device=str(self.device))
            model.load_state_dict(sd)
            self._posenet = model
        return self._posenet

    @property
    def segnet(self):
        if self._segnet is None:
            model = SegNet().eval().to(self.device)
            sd = load_file(self.segnet_weights, device=str(self.device))
            model.load_state_dict(sd)
            self._segnet = model
        return self._segnet

    def to(self, device):
        self.device = torch.device(device)
        if self._posenet is not None:
            self._posenet = self._posenet.to(self.device)
        if self._segnet is not None:
            self._segnet = self._segnet.to(self.device)
        return self

_manager = None

def get_manager(device=None, posenet_weights=None, segnet_weights=None):
    global _manager
    if _manager is None:
        _manager = EvaluatorManager(device, posenet_weights, segnet_weights)
    else:
        if device is not None:
            _manager.to(device)
        if posenet_weights is not None:
            _manager.posenet_weights = posenet_weights
            _manager._posenet = None
        if segnet_weights is not None:
            _manager.segnet_weights = segnet_weights
            _manager._segnet = None
    return _manager

def prepare_input(tensor, expected_dims):
    """
    Converts input to torch.Tensor, moves/rearranges dimensions to match
    channels-first (B, C, H, W) or (B, T, C, H, W) format, and handles floats.
    """
    if not isinstance(tensor, torch.Tensor):
        if hasattr(tensor, "numpy"):
            tensor = torch.as_tensor(tensor)
        else:
            import numpy as np
            tensor = torch.from_numpy(np.asarray(tensor))
            
    tensor = tensor.float()

    if expected_dims == 4:
        if tensor.ndim == 5:
            # Input is (B, T, H, W, C) or (B, T, C, H, W)
            if tensor.shape[-1] == 3:
                # Channels last: rearrange
                tensor = tensor.permute(0, 1, 4, 2, 3)
            # Take the last frame
            tensor = tensor[:, -1, ...]
        elif tensor.ndim == 4:
            # Input is (B, H, W, C) or (B, C, H, W)
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 4D or 5D input for SegNet, got shape {tensor.shape}")
            
    elif expected_dims == 5:
        if tensor.ndim == 4:
            # (2, H, W, 3) or (2, 3, H, W) or (B, H, W, 3) -> treat as single pair
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.unsqueeze(0)
            if tensor.shape[1] != 2:
                raise ValueError("PoseNet expects a sequence of length 2 (pair of consecutive frames).")
        elif tensor.ndim == 5:
            # (B, T, H, W, C) or (B, T, C, H, W)
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 1, 4, 2, 3)
            if tensor.shape[1] != 2:
                raise ValueError(f"PoseNet expects sequence length 2, got {tensor.shape[1]}")
        else:
            raise ValueError(f"Expected 4D or 5D input for PoseNet, got shape {tensor.shape}")
            
    return tensor

class ChallengeScore(float):
    """
    A float subclass representing the final score of the challenge.
    Allows accessing individual loss components.
    """
    def __new__(cls, score, segnet_dist, posenet_dist, rate):
        obj = super().__new__(cls, score)
        obj.segnet_distortion = segnet_dist
        obj.posenet_distortion = posenet_dist
        obj.rate = rate
        return obj

    def __repr__(self):
        return (f"ChallengeScore({super().__repr__()}, "
                f"segnet_dist={self.segnet_distortion:.6f}, "
                f"posenet_dist={self.posenet_distortion:.6f}, "
                f"rate={self.rate:.6f})")

def segnet_cost_term(gt, pred, device=None, weights_path=None):
    """
    Computes SegNet distortion between ground truth and predicted frames.
    Accepts:
      gt, pred: (B, H, W, 3) or (B, 3, H, W) batch of images.
                Also accepts (B, 2, H, W, 3) or (B, 2, 3, H, W) where the last frame is used.
    Returns:
      A PyTorch tensor of shape (B,) containing the distortion per sample.
    """
    mgr = get_manager(device=device, segnet_weights=weights_path)
    gt_tensor = prepare_input(gt, expected_dims=4).to(mgr.device)
    pred_tensor = prepare_input(pred, expected_dims=4).to(mgr.device)
    
    with torch.inference_mode():
        gt_in = mgr.segnet.preprocess_input(gt_tensor)
        pred_in = mgr.segnet.preprocess_input(pred_tensor)
        gt_out = mgr.segnet(gt_in)
        pred_out = mgr.segnet(pred_in)
        return mgr.segnet.compute_distortion(gt_out, pred_out)

def posenet_cost_term(gt, pred, device=None, weights_path=None):
    """
    Computes PoseNet distortion between ground truth and predicted frame sequences.
    Accepts:
      gt, pred: (B, 2, H, W, 3) or (B, 2, 3, H, W) batch of frame sequences.
                Also accepts single sequence (2, H, W, 3) or (2, 3, H, W).
    Returns:
      A PyTorch tensor of shape (B,) containing the distortion per sample.
    """
    mgr = get_manager(device=device, posenet_weights=weights_path)
    gt_tensor = prepare_input(gt, expected_dims=5).to(mgr.device)
    pred_tensor = prepare_input(pred, expected_dims=5).to(mgr.device)
    
    with torch.inference_mode():
        gt_in = mgr.posenet.preprocess_input(gt_tensor)
        pred_in = mgr.posenet.preprocess_input(pred_tensor)
        gt_out = mgr.posenet(gt_in)
        pred_out = mgr.posenet(pred_in)
        return mgr.posenet.compute_distortion(gt_out, pred_out)

def total_loss(gt, pred, rate=0.0, device=None, posenet_weights_path=None, segnet_weights_path=None):
    """
    Computes the exact challenge score:
      score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
    
    Accepts:
      gt, pred: Batches of frame sequences of shape (B, 2, H, W, 3) or (B, 2, 3, H, W).
      rate: The compression rate float (defaults to 0.0).
    
    Returns:
      A ChallengeScore object (which inherits from float) containing the score
      and individual distortion components.
    """
    mgr = get_manager(device=device, posenet_weights=posenet_weights_path, segnet_weights=segnet_weights_path)
    
    gt_posenet = prepare_input(gt, expected_dims=5).to(mgr.device)
    pred_posenet = prepare_input(pred, expected_dims=5).to(mgr.device)
    
    gt_segnet = prepare_input(gt, expected_dims=4).to(mgr.device)
    pred_segnet = prepare_input(pred, expected_dims=4).to(mgr.device)
    
    with torch.inference_mode():
        # PoseNet
        gt_p_in = mgr.posenet.preprocess_input(gt_posenet)
        pred_p_in = mgr.posenet.preprocess_input(pred_posenet)
        posenet_dist_batch = mgr.posenet.compute_distortion(mgr.posenet(gt_p_in), mgr.posenet(pred_p_in))
        
        # SegNet
        gt_s_in = mgr.segnet.preprocess_input(gt_segnet)
        pred_s_in = mgr.segnet.preprocess_input(pred_segnet)
        segnet_dist_batch = mgr.segnet.compute_distortion(mgr.segnet(gt_s_in), mgr.segnet(pred_s_in))
        
        # Average over the batch
        posenet_dist = posenet_dist_batch.mean().item()
        segnet_dist = segnet_dist_batch.mean().item()
        
        # Formula: 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
        score = 100 * segnet_dist + math.sqrt(10 * posenet_dist) + 25 * rate
        
        return ChallengeScore(score, segnet_dist, posenet_dist, rate)
