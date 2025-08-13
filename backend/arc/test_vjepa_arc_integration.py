#!/usr/bin/env python3
"""
Prueba de integración V-JEPA Contrastive con puzzles ARC
"""

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar el encoder contrastivo
import sys
sys.path.append('/app/arc')
from train_vjepa_contrastive import SimpleEncoder


class VJEPAContrastiveARC:
    """
    V-JEPA Contrastive para resolver puzzles ARC
    """
    
    def __init__(self, weights_path: str = "/app/arc/vjepa_contrastive_weights/vjepa_contrastive_final.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = self._load_encoder(weights_path)
        self.reset()
        
    def _load_encoder(self, weights_path: str):
        """Carga el encoder pre-entrenado"""
        encoder = SimpleEncoder(input_size=32, output_dim=128)
        
        if Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location=self.device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            encoder.eval()
            encoder.to(self.device)
            logger.info(f"✅ Encoder Contrastive cargado - Época: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        else:
            logger.warning(f"⚠️ No existe {weights_path}, usando encoder aleatorio")
            encoder.to(self.device)
            
        return encoder
    
    def reset(self):
        """Resetea memoria (cumple reglas ARC)"""
        self.puzzle_examples = []
        self.learned_patterns = []
        
    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        """Codifica un grid usando el encoder"""
        # Preparar grid a 32x32
        h, w = grid.shape
        padded = np.zeros((32, 32), dtype=np.float32)
        padded[:min(h, 32), :min(w, 32)] = grid[:min(h, 32), :min(w, 32)] / 10.0  # Normalizar
        
        # Convertir a tensor
        tensor = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Codificar
        with torch.no_grad():
            embedding = self.encoder(tensor)
        
        return embedding.cpu().numpy().squeeze()
    
    def analyze_examples(self, train_examples: List[Dict]) -> Dict:
        """Analiza los ejemplos de entrenamiento"""
        self.reset()
        results = {
            'pattern_type': 'unknown',
            'confidence': 0.0,
            'transformation': None
        }
        
        if not train_examples:
            return results
            
        # Codificar todos los ejemplos
        for example in train_examples:
            input_grid = np.array(example.get('input', []))
            output_grid = np.array(example.get('output', []))
            
            if input_grid.size == 0 or output_grid.size == 0:
                continue
                
            input_emb = self.encode_grid(input_grid)
            output_emb = self.encode_grid(output_grid)
            
            self.puzzle_examples.append({
                'input': input_grid,
                'output': output_grid,
                'input_emb': input_emb,
                'output_emb': output_emb,
                'transform_vector': output_emb - input_emb
            })
        
        # Analizar patrones
        if len(self.puzzle_examples) > 1:
            # Calcular consistencia de transformaciones
            transform_vectors = [ex['transform_vector'] for ex in self.puzzle_examples]
            
            # Similitud promedio entre vectores de transformación
            similarities = []
            for i in range(len(transform_vectors)):
                for j in range(i+1, len(transform_vectors)):
                    sim = self._cosine_similarity(transform_vectors[i], transform_vectors[j])
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            if avg_similarity > 0.8:
                results['pattern_type'] = 'consistent_transformation'
                results['confidence'] = avg_similarity
            elif avg_similarity > 0.5:
                results['pattern_type'] = 'partial_pattern'
                results['confidence'] = avg_similarity
            else:
                results['pattern_type'] = 'complex_pattern'
                results['confidence'] = 0.3
                
            # Guardar transformación promedio
            avg_transform = np.mean(transform_vectors, axis=0)
            results['transformation'] = avg_transform
            
        return results
    
    def solve(self, test_input: np.ndarray) -> np.ndarray:
        """Intenta resolver el puzzle"""
        if not self.puzzle_examples:
            return test_input.copy()
            
        # Codificar test input
        test_emb = self.encode_grid(test_input)
        
        # Encontrar ejemplo más similar
        best_match = None
        best_similarity = -1
        
        for example in self.puzzle_examples:
            sim = self._cosine_similarity(test_emb, example['input_emb'])
            if sim > best_similarity:
                best_similarity = sim
                best_match = example
        
        if best_match and best_similarity > 0.3:
            # Aplicar transformación similar
            if test_input.shape == best_match['input'].shape:
                # Intentar transformación directa
                return self._apply_transformation(
                    test_input,
                    best_match['input'],
                    best_match['output']
                )
            else:
                # Aplicar mapeo de colores
                return self._apply_color_mapping(
                    test_input,
                    best_match['input'],
                    best_match['output']
                )
        
        return test_input.copy()
    
    def _apply_transformation(self, test_grid, ref_input, ref_output):
        """Aplica transformación detectada"""
        # Detectar transformaciones geométricas simples
        if ref_input.shape == ref_output.shape:
            # Rotaciones
            for k in range(4):
                if np.array_equal(np.rot90(ref_input, k), ref_output):
                    return np.rot90(test_grid, k)
            
            # Flips
            if np.array_equal(np.fliplr(ref_input), ref_output):
                return np.fliplr(test_grid)
            if np.array_equal(np.flipud(ref_input), ref_output):
                return np.flipud(test_grid)
        
        # Si no es geométrica, intentar mapeo de colores
        return self._apply_color_mapping(test_grid, ref_input, ref_output)
    
    def _apply_color_mapping(self, test_grid, ref_input, ref_output):
        """Aplica mapeo de colores"""
        result = test_grid.copy()
        
        # Aprender mapeo
        color_map = {}
        for color in np.unique(ref_input):
            mask = ref_input == color
            if np.any(mask) and mask.shape == ref_output.shape:
                output_colors = ref_output[mask]
                if len(output_colors) > 0:
                    unique, counts = np.unique(output_colors, return_counts=True)
                    color_map[color] = unique[np.argmax(counts)]
        
        # Aplicar
        for old_color, new_color in color_map.items():
            result[test_grid == old_color] = new_color
        
        return result
    
    def _cosine_similarity(self, a, b):
        """Calcula similitud coseno"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)


def test_on_arc_puzzles():
    """Prueba el modelo con puzzles ARC reales"""
    logger.info("=== PRUEBA V-JEPA CONTRASTIVE CON PUZZLES ARC ===")
    
    # Cargar modelo
    solver = VJEPAContrastiveARC()
    
    # Cargar puzzles desde el caché
    cache_dir = Path("/app/arc_official_cache")
    puzzle_files = list(cache_dir.glob("arc_agi_1_training_*.json"))
    
    if not puzzle_files:
        logger.error("No se encontraron puzzles ARC en caché")
        return
    
    puzzles = {}
    for puzzle_file in puzzle_files[:20]:  # Cargar primeros 20
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
            puzzle_id = puzzle_file.stem.replace("arc_agi_1_training_", "")
            puzzles[puzzle_id] = puzzle_data
    
    # Seleccionar algunos puzzles para probar
    test_puzzles = list(puzzles.items())[:10]  # Primeros 10
    
    results = []
    logger.info(f"\nProbando con {len(test_puzzles)} puzzles:")
    
    for puzzle_id, puzzle_data in test_puzzles:
        logger.info(f"\n--- Puzzle: {puzzle_id} ---")
        
        # Resetear para cada puzzle (regla ARC)
        solver.reset()
        
        # Analizar ejemplos de entrenamiento
        train_examples = puzzle_data.get('train', [])
        analysis = solver.analyze_examples(train_examples)
        
        logger.info(f"  Patrón detectado: {analysis['pattern_type']}")
        logger.info(f"  Confianza: {analysis['confidence']:.3f}")
        
        # Intentar resolver test
        test_cases = puzzle_data.get('test', [])
        if test_cases:
            test_input = np.array(test_cases[0]['input'])
            prediction = solver.solve(test_input)
            
            # Verificar si hay respuesta correcta
            if 'output' in test_cases[0]:
                correct_output = np.array(test_cases[0]['output'])
                is_correct = np.array_equal(prediction, correct_output)
                
                # Calcular precisión solo si las dimensiones coinciden
                if prediction.shape == correct_output.shape:
                    accuracy = np.mean(prediction == correct_output)
                else:
                    accuracy = 0.0
                    logger.info(f"  Dimensiones incorrectas: {prediction.shape} vs {correct_output.shape}")
                
                logger.info(f"  Correcto: {'✅' if is_correct else '❌'}")
                if prediction.shape == correct_output.shape:
                    logger.info(f"  Precisión pixel: {accuracy:.1%}")
                
                results.append({
                    'puzzle_id': puzzle_id,
                    'pattern': analysis['pattern_type'],
                    'confidence': analysis['confidence'],
                    'correct': is_correct,
                    'accuracy': accuracy
                })
            else:
                logger.info("  (Sin respuesta para verificar)")
    
    # Resumen
    if results:
        logger.info("\n=== RESUMEN DE RESULTADOS ===")
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        logger.info(f"Puzzles correctos: {correct_count}/{total_count} ({correct_count/total_count:.1%})")
        logger.info(f"Precisión promedio: {avg_accuracy:.1%}")
        logger.info(f"Confianza promedio: {avg_confidence:.3f}")
        
        # Análisis por tipo de patrón
        pattern_types = {}
        for r in results:
            pattern = r['pattern']
            if pattern not in pattern_types:
                pattern_types[pattern] = []
            pattern_types[pattern].append(r['correct'])
        
        logger.info("\nPor tipo de patrón:")
        for pattern, corrects in pattern_types.items():
            success_rate = sum(corrects) / len(corrects) if corrects else 0
            logger.info(f"  {pattern}: {success_rate:.1%} ({sum(corrects)}/{len(corrects)})")
    
    return results


if __name__ == "__main__":
    test_on_arc_puzzles()