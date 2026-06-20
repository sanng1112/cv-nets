import argparse
import torch

from loss_fn import LOSS_REGISTRY

def test_classification_losses():
    print("--- Testing Classification Losses ---")
    losses = ["cross_entropy", "focal_loss", "bce", "kld", "poly_loss"]
    
    for loss_name in losses:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "classification")
        setattr(opts, "loss.classification.name", loss_name)
        
        # Add required defaults
        if loss_name == "focal_loss":
            setattr(opts, "loss.classification.focal_loss.gamma", 2.0)
            setattr(opts, "loss.classification.focal_loss.alpha", 0.25)
            setattr(opts, "loss.classification.focal_loss.ignore_index", -1)
        elif loss_name == "cross_entropy":
            setattr(opts, "loss.classification.cross_entropy.ignore_index", -1)
            setattr(opts, "loss.classification.cross_entropy.class_weights", False)
            setattr(opts, "loss.classification.cross_entropy.label_smoothing", 0.0)
        elif loss_name == "kld":
            setattr(opts, "loss.classification.kld.log_target", False)
        elif loss_name == "poly_loss":
            setattr(opts, "loss.classification.poly_loss.epsilon", 2.0)
            setattr(opts, "loss.classification.poly_loss.ignore_index", -1)
            
        try:
            criterion = LOSS_REGISTRY[loss_name, "classification"](opts)
            # Create dummy data
            B, C = 2, 5
            pred = torch.randn(B, C, requires_grad=True)
            if loss_name == "bce":
                target = torch.randint(0, C, (B,))
                loss = criterion(input_sample=None, prediction=pred, target=target)
            elif loss_name == "kld":
                target = torch.randint(0, C, (B,))
                loss = criterion(input_sample=None, prediction=pred, target=target)
            else:
                target = torch.randint(0, C, (B,))
                loss = criterion(input_sample=None, prediction=pred, target=target)
            
            loss.backward()
            print(f"[{loss_name}] \tOK! Loss value: {loss.item():.4f}")
        except Exception as e:
            print(f"[{loss_name}] \tFAILED! Error: {e}")

def test_detection_losses():
    print("\n--- Testing Detection Losses ---")
    losses = ["smooth_l1", "giou", "diou", "ciou"]
    
    for loss_name in losses:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "detection")
        setattr(opts, "loss.detection.name", loss_name)
        
        if loss_name == "smooth_l1":
            setattr(opts, "loss.detection.smooth_l1.beta", 1.0)
        else:
            setattr(opts, f"loss.detection.{loss_name}.reduction", "mean")
            
        try:
            criterion = LOSS_REGISTRY[loss_name, "detection"](opts)
            # Create dummy data (boxes)
            # Format: [x1, y1, x2, y2]
            pred = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 40.0, 40.0]], requires_grad=True)
            target = torch.tensor([[15.0, 15.0, 45.0, 45.0], [20.0, 20.0, 40.0, 40.0]])
            
            loss = criterion(input_sample=None, prediction=pred, target=target)
            loss.backward()
            print(f"[{loss_name}] \tOK! Loss value: {loss.item():.4f}")
        except Exception as e:
            print(f"[{loss_name}] \tFAILED! Error: {e}")

def test_segmentation_losses():
    print("\n--- Testing Segmentation Losses ---")
    losses = ["dice", "jaccard", "tversky", "focal_tversky"]
    
    for loss_name in losses:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "segmentation")
        setattr(opts, "loss.segmentation.name", loss_name)
        
        # defaults
        if loss_name == "dice":
            setattr(opts, "loss.segmentation.dice.smooth", 1.0)
        elif loss_name == "jaccard":
            setattr(opts, "loss.segmentation.jaccard.smooth", 1.0)
        elif loss_name == "tversky":
            setattr(opts, "loss.segmentation.tversky.smooth", 1.0)
            setattr(opts, "loss.segmentation.tversky.alpha", 0.5)
            setattr(opts, "loss.segmentation.tversky.beta", 0.5)
        elif loss_name == "focal_tversky":
            setattr(opts, "loss.segmentation.focal_tversky.smooth", 1.0)
            setattr(opts, "loss.segmentation.focal_tversky.alpha", 0.5)
            setattr(opts, "loss.segmentation.focal_tversky.beta", 0.5)
            setattr(opts, "loss.segmentation.focal_tversky.gamma", 0.75)
            
        try:
            criterion = LOSS_REGISTRY[loss_name, "segmentation"](opts)
            # Create dummy data (masks)
            B, C, H, W = 2, 3, 32, 32
            pred = torch.randn(B, C, H, W, requires_grad=True)
            target = torch.randint(0, C, (B, H, W))
            
            loss = criterion(input_sample=None, prediction=pred, target=target)
            loss.backward()
            print(f"[{loss_name}] \tOK! Loss value: {loss.item():.4f}")
        except Exception as e:
            print(f"[{loss_name}] \tFAILED! Error: {e}")

def test_regression_losses():
    print("\n--- Testing Regression Losses ---")
    losses = ["l1", "l2", "log_cosh"]
    
    for loss_name in losses:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "regression")
        setattr(opts, "loss.regression.name", loss_name)
        
        # defaults
        setattr(opts, f"loss.regression.{loss_name}.reduction", "mean")
            
        try:
            criterion = LOSS_REGISTRY[loss_name, "regression"](opts)
            # Create dummy data
            B, D = 4, 10
            pred = torch.randn(B, D, requires_grad=True)
            target = torch.randn(B, D)
            
            loss = criterion(input_sample=None, prediction=pred, target=target)
            loss.backward()
            print(f"[{loss_name}] \tOK! Loss value: {loss.item():.4f}")
        except Exception as e:
            print(f"[{loss_name}] \tFAILED! Error: {e}")

def test_metric_losses():
    print("\n--- Testing Metric Learning Losses ---")
    
    # Test Triplet
    try:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "metric_learning")
        setattr(opts, "loss.metric_learning.name", "triplet")
        setattr(opts, "loss.metric_learning.triplet.margin", 1.0)
        setattr(opts, "loss.metric_learning.triplet.p", 2.0)
        criterion = LOSS_REGISTRY["triplet", "metric_learning"](opts)
        
        anchor = torch.randn(2, 128, requires_grad=True)
        positive = torch.randn(2, 128, requires_grad=True)
        negative = torch.randn(2, 128, requires_grad=True)
        
        loss = criterion(input_sample=None, prediction=(anchor, positive, negative), target=None)
        loss.backward()
        print(f"[triplet] \tOK! Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"[triplet] \tFAILED! Error: {e}")
        
    # Test Contrastive
    try:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "metric_learning")
        setattr(opts, "loss.metric_learning.name", "contrastive")
        setattr(opts, "loss.metric_learning.contrastive.margin", 1.0)
        criterion = LOSS_REGISTRY["contrastive", "metric_learning"](opts)
        
        out1 = torch.randn(2, 128, requires_grad=True)
        out2 = torch.randn(2, 128, requires_grad=True)
        target = torch.randint(0, 2, (2, 1)).float()
        
        loss = criterion(input_sample=None, prediction=(out1, out2), target=target)
        loss.backward()
        print(f"[contrastive] \tOK! Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"[contrastive] \tFAILED! Error: {e}")

def test_generative_losses():
    print("\n--- Testing Generative / Image Restoration Losses ---")
    
    losses = ["psnr", "tv", "ssim"]
    
    for loss_name in losses:
        opts = argparse.Namespace()
        setattr(opts, "loss.category", "generative")
        setattr(opts, "loss.generative.name", loss_name)
        
        if loss_name == "psnr":
            setattr(opts, "loss.generative.psnr.max_val", 1.0)
        elif loss_name == "tv":
            setattr(opts, "loss.generative.tv.weight", 1.0)
        elif loss_name == "ssim":
            setattr(opts, "loss.generative.ssim.window_size", 11)
            
        try:
            criterion = LOSS_REGISTRY[loss_name, "generative"](opts)
            B, C, H, W = 2, 3, 32, 32
            pred = torch.randn(B, C, H, W, requires_grad=True)
            target = torch.randn(B, C, H, W)
            
            # TV loss doesn't require target strictly but base criteria allows None
            if loss_name == "tv":
                loss = criterion(input_sample=None, prediction=pred, target=None)
            else:
                loss = criterion(input_sample=None, prediction=pred, target=target)
                
            loss.backward()
            print(f"[{loss_name}] \tOK! Loss value: {loss.item():.4f}")
        except Exception as e:
            print(f"[{loss_name}] \tFAILED! Error: {e}")

if __name__ == "__main__":
    test_classification_losses()
    test_detection_losses()
    test_segmentation_losses()
    test_regression_losses()
    test_metric_losses()
    test_generative_losses()
