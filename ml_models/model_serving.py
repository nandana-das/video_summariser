"""
ML model serving and inference optimization for video summarization.
"""
import torch
import torch.nn as nn
import numpy as np
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. ONNX features will be disabled.")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import mlflow
import mlflow.pytorch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class InferenceRequest(BaseModel):
    """Request model for inference API."""
    text: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 1.0
    num_beams: Optional[int] = 4
    do_sample: Optional[bool] = False

class InferenceResponse(BaseModel):
    """Response model for inference API."""
    summary: str
    processing_time: float
    model_name: str
    metadata: Dict[str, Any]

class ModelOptimizer:
    """Model optimization for faster inference."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize model optimizer.
        
        Args:
            device: Device for optimization
        """
        self.device = self._get_device(device)
        self.optimized_models = {}
        
        logger.info(f"ModelOptimizer initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def optimize_model(self, model: nn.Module, tokenizer,
                      optimization_type: str = "quantization",
                      save_path: Optional[Union[str, Path]] = None) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            tokenizer: Tokenizer for the model
            optimization_type: Type of optimization ('quantization', 'pruning', 'onnx')
            save_path: Path to save optimized model
            
        Returns:
            Optimized model
        """
        try:
            logger.info(f"Optimizing model with {optimization_type}")
            
            if optimization_type == "quantization":
                optimized_model = self._quantize_model(model)
            elif optimization_type == "pruning":
                optimized_model = self._prune_model(model)
            elif optimization_type == "onnx":
                optimized_model = self._convert_to_onnx(model, tokenizer, save_path)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            if save_path:
                self._save_optimized_model(optimized_model, save_path, optimization_type)
            
            logger.info(f"Model optimization completed: {optimization_type}")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to the model."""
        try:
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error quantizing model: {str(e)}")
            raise
    
    def _prune_model(self, model: nn.Module, sparsity: float = 0.1) -> nn.Module:
        """Apply pruning to the model."""
        try:
            import torch.nn.utils.prune as prune
            
            # Prune linear layers
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
            
            return model
            
        except Exception as e:
            logger.error(f"Error pruning model: {str(e)}")
            raise
    
    def _convert_to_onnx(self, model: nn.Module, tokenizer, 
                        save_path: Optional[Union[str, Path]] = None) -> nn.Module:
        """Convert model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is not available. Please install onnx and onnxruntime.")
        
        try:
            model.eval()
            
            # Create dummy input
            dummy_input = tokenizer(
                "This is a dummy input for ONNX conversion.",
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True
            )
            
            # Convert to ONNX
            if save_path is None:
                save_path = "model.onnx"
            
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'output': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            logger.info(f"Model converted to ONNX: {save_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error converting to ONNX: {str(e)}")
            raise
    
    def _save_optimized_model(self, model: nn.Module, save_path: Union[str, Path],
                             optimization_type: str):
        """Save optimized model."""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            if optimization_type == "onnx":
                # ONNX model is already saved
                pass
            else:
                # Save PyTorch model
                torch.save(model.state_dict(), save_path / "optimized_model.pt")
            
            # Save optimization metadata
            metadata = {
                "optimization_type": optimization_type,
                "device": self.device,
                "timestamp": time.time()
            }
            
            with open(save_path / "optimization_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Optimized model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving optimized model: {str(e)}")
            raise

class ONNXInferenceEngine:
    """ONNX-based inference engine for optimized inference."""
    
    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        """
        Initialize ONNX inference engine.
        
        Args:
            model_path: Path to ONNX model
            device: Device for inference
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is not available. Please install onnx and onnxruntime.")
        
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.session = None
        
        self._load_model()
        logger.info(f"ONNXInferenceEngine initialized with {model_path}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load ONNX model."""
        try:
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            logger.info("ONNX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Model predictions
        """
        try:
            inputs = {
                'input_ids': input_ids.astype(np.int64),
                'attention_mask': attention_mask.astype(np.int64)
            }
            
            outputs = self.session.run(None, inputs)
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Error running ONNX inference: {str(e)}")
            raise

class InferenceServer:
    """High-performance inference server."""
    
    def __init__(self, model_path: Union[str, Path], 
                 max_workers: int = 4,
                 queue_size: int = 100):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to trained model
            max_workers: Maximum number of worker threads
            queue_size: Maximum queue size
        """
        self.model_path = Path(model_path)
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = ModelOptimizer()
        self.request_queue = queue.Queue(maxsize=queue_size)
        self.result_cache = {}
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Load model
        self._load_model()
        
        logger.info(f"InferenceServer initialized with {max_workers} workers")
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Optimize model for inference
            self.model = self.optimizer.optimize_model(
                self.model, self.tokenizer, "quantization"
            )
            
            self.model.eval()
            logger.info("Model loaded and optimized for inference")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str, max_length: int = 150,
               temperature: float = 1.0, num_beams: int = 4,
               do_sample: bool = False) -> Dict[str, Any]:
        """
        Generate prediction for input text.
        
        Args:
            text: Input text
            max_length: Maximum length of output
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            Prediction results
        """
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return {
                'summary': summary,
                'processing_time': processing_time,
                'model_name': str(self.model_path),
                'metadata': {
                    'input_length': len(text.split()),
                    'output_length': len(summary.split()),
                    'compression_ratio': len(summary.split()) / len(text.split()) if len(text.split()) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise
    
    def batch_predict(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate predictions for multiple texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional prediction parameters
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            # Process in parallel
            futures = []
            for text in texts:
                future = self.executor.submit(self.predict, text, **kwargs)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch prediction: {str(e)}")
                    results.append({
                        'summary': '',
                        'processing_time': 0,
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def async_predict(self, text: str, **kwargs) -> asyncio.Task:
        """
        Generate prediction asynchronously.
        
        Args:
            text: Input text
            **kwargs: Additional prediction parameters
            
        Returns:
            Async task for prediction
        """
        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(self.executor, self.predict, text, **kwargs)
            return future
            
        except Exception as e:
            logger.error(f"Error creating async prediction: {str(e)}")
            raise

class ModelServingAPI:
    """FastAPI-based model serving API."""
    
    def __init__(self, model_path: Union[str, Path], 
                 host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize model serving API.
        
        Args:
            model_path: Path to trained model
            host: API host
            port: API port
        """
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Video Summarization API",
            description="ML-powered video summarization service",
            version="1.0.0"
        )
        
        # Initialize inference server
        self.inference_server = InferenceServer(model_path)
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"ModelServingAPI initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Video Summarization API", "status": "running"}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest):
            try:
                result = self.inference_server.predict(
                    text=request.text,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    num_beams=request.num_beams,
                    do_sample=request.do_sample
                )
                
                return InferenceResponse(
                    summary=result['summary'],
                    processing_time=result['processing_time'],
                    model_name=result['model_name'],
                    metadata=result['metadata']
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_predict")
        async def batch_predict(requests: List[InferenceRequest]):
            try:
                texts = [req.text for req in requests]
                results = self.inference_server.batch_predict(texts)
                
                return {"results": results}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model_info")
        async def model_info():
            try:
                return {
                    "model_path": str(self.model_path),
                    "max_workers": self.inference_server.max_workers,
                    "queue_size": self.inference_server.queue_size
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, reload: bool = False):
        """Run the API server."""
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                reload=reload
            )
            
        except Exception as e:
            logger.error(f"Error running API server: {str(e)}")
            raise

class ModelRegistry:
    """Model registry for managing multiple models."""
    
    def __init__(self, registry_path: Union[str, Path] = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.active_model = None
        
        # Load existing models
        self._load_registry()
        
        logger.info(f"ModelRegistry initialized at {registry_path}")
    
    def _load_registry(self):
        """Load existing model registry."""
        try:
            registry_file = self.registry_path / "registry.json"
            
            if registry_file.exists():
                with open(registry_file, "r") as f:
                    self.models = json.load(f)
                
                logger.info(f"Loaded {len(self.models)} models from registry")
            
        except Exception as e:
            logger.warning(f"Error loading registry: {str(e)}")
    
    def register_model(self, model_name: str, model_path: Union[str, Path],
                      description: str = "", tags: List[str] = None) -> str:
        """
        Register a new model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model
            description: Model description
            tags: Model tags
            
        Returns:
            Model ID
        """
        try:
            model_id = f"{model_name}_{int(time.time())}"
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
            
            model_info = {
                "model_id": model_id,
                "model_name": model_name,
                "model_path": str(model_path),
                "description": description,
                "tags": tags or [],
                "created_at": time.time(),
                "status": "registered"
            }
            
            self.models[model_id] = model_info
            self._save_registry()
            
            logger.info(f"Model registered: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def set_active_model(self, model_id: str):
        """Set the active model."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            self.active_model = model_id
            logger.info(f"Active model set to: {model_id}")
            
        except Exception as e:
            logger.error(f"Error setting active model: {str(e)}")
            raise
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            return self.models[model_id]
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return list(self.models.values())
    
    def _save_registry(self):
        """Save model registry."""
        try:
            registry_file = self.registry_path / "registry.json"
            
            with open(registry_file, "w") as f:
                json.dump(self.models, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
            raise

def main():
    """Example usage of the model serving components."""
    # Initialize model registry
    registry = ModelRegistry()
    
    # Register a model
    model_id = registry.register_model(
        "bart-summarizer",
        "path/to/model",
        "BART model for text summarization",
        ["summarization", "bart"]
    )
    
    # Set as active model
    registry.set_active_model(model_id)
    
    # Initialize inference server
    inference_server = InferenceServer("path/to/model")
    
    # Generate prediction
    result = inference_server.predict("This is a sample text for summarization.")
    print(f"Summary: {result['summary']}")
    print(f"Processing time: {result['processing_time']:.2f}s")

if __name__ == "__main__":
    main()
