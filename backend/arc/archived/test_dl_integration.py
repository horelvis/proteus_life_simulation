#!/usr/bin/env python3
"""
Script de prueba para verificar la integración del sistema Deep Learning corregido
"""

import numpy as np
import torch
import logging
from deep_learning_solver_fixed import DeepLearningARCSolver
from dl_training_integrated import ARCDataset, ARCCollator, ARCTrainerIntegrated, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_forward():
    """Prueba el forward pass del modelo"""
    logger.info("=== Test Forward Pass ===")
    
    # Crear modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepLearningARCSolver(device=device)
    
    # Crear datos de prueba
    batch_size = 2
    input_tensor = torch.randn(batch_size, 10, 15, 15).to(device)
    target_tensor = torch.randn(batch_size, 10, 15, 15).to(device)
    
    # Forward pass
    try:
        outputs = model(input_tensor, target_tensor)
        
        logger.info(f"✓ Forward pass exitoso")
        logger.info(f"  - Input embeds shape: {outputs['input_embeds'].shape}")
        logger.info(f"  - Logits shape: {outputs['logits'].shape}")
        logger.info(f"  - Value shape: {outputs['value'].shape}")
        logger.info(f"  - Transformation shape: {outputs['transformation'].shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error en forward pass: {e}")
        return False

def test_gradient_flow():
    """Prueba que los gradientes fluyan correctamente"""
    logger.info("\n=== Test Gradient Flow ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepLearningARCSolver(device=device)
    
    # Crear datos de prueba
    input_tensor = torch.randn(1, 10, 10, 10, requires_grad=True).to(device)
    target_tensor = torch.randn(1, 10, 10, 10).to(device)
    
    # Forward pass
    outputs = model(input_tensor, target_tensor)
    
    # Calcular pérdida simple
    loss = outputs['logits'].mean() + outputs['value'].mean()
    
    # Backward pass
    try:
        loss.backward()
        
        # Verificar gradientes
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        if has_gradients:
            logger.info(f"✓ Gradientes fluyen correctamente")
            return True
        else:
            logger.error(f"✗ No se detectaron gradientes")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error en backward pass: {e}")
        return False

def test_inference():
    """Prueba el modo de inferencia"""
    logger.info("\n=== Test Inference ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepLearningARCSolver(device=device)
    model.eval()
    
    # Crear puzzle de prueba
    train_examples = [
        {
            'input': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            'output': [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        }
    ]
    
    test_input = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    
    try:
        with torch.no_grad():
            solution = model.solve(train_examples, test_input)
        
        logger.info(f"✓ Inferencia exitosa")
        logger.info(f"  - Input shape: {test_input.shape}")
        logger.info(f"  - Output shape: {solution.shape}")
        logger.info(f"  - Input:\n{test_input}")
        logger.info(f"  - Solution:\n{solution}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error en inferencia: {e}")
        return False

def test_dataset_loading():
    """Prueba la carga del dataset"""
    logger.info("\n=== Test Dataset Loading ===")
    
    # Intentar cargar dataset
    data_path = "/app/arc/arc_official_cache"
    
    try:
        dataset = ARCDataset(data_path, max_size=30)
        
        if len(dataset) > 0:
            logger.info(f"✓ Dataset cargado: {len(dataset)} puzzles")
            
            # Probar acceso a un elemento
            sample = dataset[0]
            logger.info(f"  - Primer puzzle ID: {sample['id']}")
            logger.info(f"  - Input shape: {sample['input'].shape}")
            logger.info(f"  - Output shape: {sample['output'].shape}")
            
            return True
        else:
            logger.warning(f"⚠ Dataset vacío en {data_path}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error cargando dataset: {e}")
        return False

def test_training_step():
    """Prueba un paso de entrenamiento"""
    logger.info("\n=== Test Training Step ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuración mínima
    config = TrainingConfig(
        batch_size=2,
        num_epochs=1,
        device=device
    )
    
    # Crear modelo
    model = DeepLearningARCSolver(device=device)
    
    # Crear datos sintéticos
    batch = {
        'input': torch.randn(2, 10, 30, 30).to(device),
        'output': torch.randn(2, 10, 30, 30).to(device),
        'mask': torch.ones(2, 30, 30).to(device)
    }
    
    # Crear optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    try:
        # Forward
        model.train()
        outputs = model(batch['input'], batch['output'])
        
        # Loss (simplificada)
        loss = outputs['logits'].mean() + outputs['value'].mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✓ Paso de entrenamiento exitoso")
        logger.info(f"  - Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en paso de entrenamiento: {e}")
        return False

def test_full_integration():
    """Prueba la integración completa del sistema"""
    logger.info("\n=== Test Integración Completa ===")
    
    # Ejecutar todas las pruebas
    tests = [
        ("Forward Pass", test_model_forward),
        ("Gradient Flow", test_gradient_flow),
        ("Inference", test_inference),
        ("Dataset Loading", test_dataset_loading),
        ("Training Step", test_training_step)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Resumen
    logger.info("\n" + "="*50)
    logger.info("RESUMEN DE PRUEBAS:")
    logger.info("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*50)
    
    if all_passed:
        logger.info("🎉 TODAS LAS PRUEBAS PASARON - Sistema integrado correctamente")
    else:
        logger.warning("⚠️  Algunas pruebas fallaron - Revisar implementación")
    
    return all_passed

if __name__ == "__main__":
    test_full_integration()