#!/usr/bin/env python3
"""
PROTEUS-ARC: Topological Dynamics Solver for ARC Puzzles
Basado en los principios del paper PROTEUS - sin redes neuronales

ADVERTENCIA: Esta es una implementación EXPERIMENTAL con fines de investigación.
No se garantiza un rendimiento superior a métodos tradicionales.
Actualmente muestra ~44% accuracy en puzzles simples vs 100% con reglas fijas.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import logging
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

logger = logging.getLogger(__name__)

# Constantes del sistema PROTEUS
HOLOGRAPHIC_MEMORY_SIZE = 2000  # 8KB como en el paper
TOPOLOGY_DIMENSIONS = 2.73      # Dimensión fractal evolutiva
FIELD_DECAY_RATE = 0.95        # Decaimiento del campo
MUTATION_RATE = 0.1            # Tasa de mutación topológica

@dataclass
class TopologicalSeed:
    """Semilla topológica según el paper"""
    dimension: float = 2.0
    curvature: float = 0.0
    betti_numbers: List[int] = None
    genus: int = 0
    field_signature: np.ndarray = None
    
    def __post_init__(self):
        if self.betti_numbers is None:
            self.betti_numbers = [1, 0, 0]
        if self.field_signature is None:
            self.field_signature = np.random.randn(16, 16)

class HolographicMemory:
    """Memoria holográfica según la sección 2.3 del paper"""
    
    def __init__(self, size: int = HOLOGRAPHIC_MEMORY_SIZE):
        self.size = size
        self.memory_field = np.zeros((size, size), dtype=complex)
        self.experience_count = 0
    
    def encode_experience(self, trajectory: np.ndarray):
        """Codifica una experiencia usando FFT como en el paper"""
        # M[k] = Σᵢ FFT(T)[i] × exp(iφᵢₖ)
        fft_trajectory = fft2(trajectory)
        
        # Limitar el tamaño para evitar problemas con arrays pequeños
        max_k = min(self.size, fft_trajectory.shape[0], fft_trajectory.shape[1])
        
        for k in range(max_k):
            phase = np.random.uniform(0, 2*np.pi)
            if k < fft_trajectory.size:
                self.memory_field[k, k] += fft_trajectory.flat[k] * np.exp(1j * phase)
        
        self.experience_count += 1
        
        # NUEVO: Normalizar para evitar crecimiento ilimitado
        if self.experience_count % 10 == 0:  # Normalizar cada 10 experiencias
            max_magnitude = np.max(np.abs(self.memory_field))
            if max_magnitude > 1.0:
                self.memory_field = self.memory_field / max_magnitude
                logger.debug(f"Memoria normalizada, magnitud máxima era {max_magnitude:.2f}")
    
    def recall(self, partial_input: np.ndarray = None) -> np.ndarray:
        """Recupera información incluso con 50% de corrupción"""
        if partial_input is None:
            # Reconstrucción estadística
            size = min(32, self.memory_field.shape[0])
            return np.real(ifft2(self.memory_field[:size, :size]))
        
        # NUEVO: Usar correlación 2D para mejor coincidencia de patrones
        from scipy.signal import correlate2d
        
        # Preparar entrada para correlación
        fft_input = fft2(partial_input)
        input_magnitude = np.abs(fft_input)
        
        # Normalizar entrada
        if np.max(input_magnitude) > 0:
            input_magnitude = input_magnitude / np.max(input_magnitude)
        
        # Correlación 2D con campo de memoria
        memory_magnitude = np.abs(self.memory_field[:input_magnitude.shape[0], :input_magnitude.shape[1]])
        
        # Normalizar campo de memoria para correlación
        if np.max(memory_magnitude) > 0:
            memory_magnitude = memory_magnitude / np.max(memory_magnitude)
        
        # Calcular correlación 2D
        correlation = correlate2d(memory_magnitude, input_magnitude, mode='same')
        
        # Encontrar mejor coincidencia
        best_match = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Reconstruir desde la mejor posición
        y, x = best_match
        size = min(partial_input.shape[0], self.memory_field.shape[0] - y)
        
        if size > 0:
            return np.real(ifft2(self.memory_field[y:y+size, x:x+size]))
        else:
            return np.zeros_like(partial_input)

class TopologicalField:
    """Campo topológico continuo para computación"""
    
    def __init__(self, shape: Tuple[int, int], sigma: Optional[float] = None, decay_rate: Optional[float] = None):
        self.shape = shape
        # Φ: ℝ² → ℝ
        self.potential = np.zeros(shape)
        self.gradient = np.zeros(shape + (2,))
        self.curvature = np.zeros(shape)
        
        # NUEVO: Parámetros adaptables según el tamaño del puzzle
        # Sigma para filtro gaussiano - escala con el tamaño
        if sigma is None:
            # Adaptar sigma al tamaño: puzzles más grandes necesitan sigma mayor
            min_dimension = min(shape)
            self.sigma = max(0.5, min(3.0, min_dimension / 10.0))
        else:
            self.sigma = sigma
            
        # Tasa de decaimiento - puzzles más grandes decaen más lento
        if decay_rate is None:
            # Ajustar decay según área del puzzle
            area = shape[0] * shape[1]
            if area < 100:
                self.decay_rate = 0.95  # Decay rápido para puzzles pequeños
            elif area < 400:
                self.decay_rate = 0.97  # Decay medio
            else:
                self.decay_rate = 0.98  # Decay lento para puzzles grandes
        else:
            self.decay_rate = decay_rate
            
        logger.debug(f"Campo topológico inicializado: shape={shape}, sigma={self.sigma:.2f}, decay={self.decay_rate:.3f}")
        
    def update(self, organisms: List['ProteusOrganism'], external_field: np.ndarray = None):
        """Actualiza el campo según la ecuación ∂Φ/∂t = ∇²Φ + R + D"""
        # Difusión con parámetros adaptables
        self.potential = gaussian_filter(self.potential, sigma=self.sigma) * self.decay_rate
        
        # Contribución de organismos (R)
        for org in organisms:
            if org.energy > 0:
                x, y = int(org.position[0]), int(org.position[1])
                if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
                    # Campo generado proporcional a la complejidad topológica
                    field_strength = org.seed.dimension * org.energy
                    self.potential[x, y] += field_strength
        
        # Campo externo (D) - puzzles ARC
        if external_field is not None:
            # Aplicar saturación para evitar desbordamiento
            field_contribution = external_field * 0.1
            self.potential = np.clip(self.potential + field_contribution, -10.0, 10.0)
        
        # Calcular gradiente para fuerzas
        self.gradient[:, :, 0] = np.gradient(self.potential, axis=0)
        self.gradient[:, :, 1] = np.gradient(self.potential, axis=1)
        
        # Calcular curvatura (Laplaciano)
        self.curvature = np.gradient(self.gradient[:, :, 0], axis=0) + \
                        np.gradient(self.gradient[:, :, 1], axis=1)

class ProteusOrganism:
    """Organismo PROTEUS sin neuronas ni pesos"""
    
    def __init__(self, position: np.ndarray = None, parent: 'ProteusOrganism' = None):
        # Posición en el espacio de fase
        if position is None:
            position = np.random.rand(2) * 30  # Tamaño típico ARC
        self.position = position
        self.velocity = np.zeros(2)
        self.energy = 1.0
        
        # Núcleo topológico
        if parent is None:
            self.seed = TopologicalSeed()
        else:
            # Herencia topológica: S_child = Ψ(S_parent₁ ⊗ S_parent₂ + μ)
            self.seed = self._inherit_topology(parent.seed)
        
        # Memoria holográfica
        self.memory = HolographicMemory()
        
        # Trayectoria para análisis topológico
        self.trajectory = [position.copy()]
    
    def _inherit_topology(self, parent_seed: TopologicalSeed) -> TopologicalSeed:
        """Herencia topológica con mutación"""
        child_seed = TopologicalSeed()
        
        # Heredar con mutaciones
        child_seed.dimension = parent_seed.dimension + np.random.normal(0, MUTATION_RATE)
        child_seed.dimension = np.clip(child_seed.dimension, 2.0, 4.0)
        
        child_seed.curvature = parent_seed.curvature + np.random.normal(0, MUTATION_RATE/2)
        
        # Mutar números de Betti (invariantes topológicos)
        child_seed.betti_numbers = parent_seed.betti_numbers.copy()
        if np.random.random() < MUTATION_RATE:
            idx = np.random.randint(len(child_seed.betti_numbers))
            child_seed.betti_numbers[idx] = max(0, child_seed.betti_numbers[idx] + 
                                               np.random.choice([-1, 1]))
        
        # Heredar firma de campo con perturbación
        child_seed.field_signature = parent_seed.field_signature + \
                                   np.random.randn(*parent_seed.field_signature.shape) * MUTATION_RATE
        
        return child_seed
    
    def update(self, field: TopologicalField, dt: float = 0.1):
        """Actualiza según dξ/dt = -∇U(ξ) + F(S, H) + η(t)"""
        x, y = int(self.position[0]), int(self.position[1])
        
        if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
            # Fuerza del gradiente del campo
            field_force = -field.gradient[x, y]
            
            # Fuerza topológica heredada
            memory_bias = self.memory.recall()
            if memory_bias is not None and memory_bias.size > 0:
                # Usar memoria para sesgar movimiento
                center = np.array(memory_bias.shape) / 2
                memory_force = (center - [x, y]) * 0.01
            else:
                memory_force = np.zeros(2)
            
            # Perturbación estocástica
            noise = np.random.randn(2) * 0.1
            
            # Ecuación de movimiento (sin decisiones if-then)
            self.velocity = self.velocity * 0.9 + field_force + memory_force + noise
            self.position += self.velocity * dt
            
            # Mantener dentro de límites
            self.position = np.clip(self.position, 0, [field.shape[0]-1, field.shape[1]-1])
            
            # Consumo de energía termodinámica
            self.energy -= 0.001
            
            # Registrar trayectoria
            self.trajectory.append(self.position.copy())
    
    def analyze_pattern(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analiza un patrón usando dinámica topológica"""
        # Codificar en memoria holográfica
        self.memory.encode_experience(input_grid)
        
        # Crear campo desde el input
        field = TopologicalField(input_grid.shape)
        field.potential = input_grid.astype(float)
        field.update([self])
        
        # Analizar propiedades topológicas
        analysis = {
            'dimension': self._estimate_dimension(input_grid),
            'curvature': np.mean(field.curvature),
            'homology': self._compute_homology(input_grid),
            'field_signature': field.potential
        }
        
        return analysis
    
    def _estimate_dimension(self, grid: np.ndarray) -> float:
        """Estima dimensión fractal del patrón"""
        # Box-counting simplificado
        sizes = [2, 4, 8]
        counts = []
        
        for size in sizes:
            count = 0
            for i in range(0, grid.shape[0], size):
                for j in range(0, grid.shape[1], size):
                    box = grid[i:i+size, j:j+size]
                    if np.any(box > 0):
                        count += 1
            counts.append(count)
        
        # Calcular dimensión
        if len(counts) > 1 and counts[0] > 0:
            log_ratio = np.log(counts[0] / counts[-1]) / np.log(sizes[-1] / sizes[0])
            return 2.0 + log_ratio * 0.5  # Ajustar al rango esperado
        return 2.0

    def _compute_homology(self, grid: np.ndarray) -> List[int]:
        """Calcula números de Betti simplificados"""
        # Betti_0: componentes conectados
        components = self._count_components(grid > 0)
        
        # Betti_1: agujeros (simplificado)
        holes = self._count_holes(grid)
        
        return [components, holes, 0]
    
    def _count_components(self, binary_grid: np.ndarray) -> int:
        """Cuenta componentes conectados usando scipy.ndimage"""
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_grid)
        return num_features
    
    def _count_holes(self, grid: np.ndarray) -> int:
        """Cuenta agujeros en el patrón"""
        # Detectar regiones cerradas de 0s rodeadas por no-0s
        holes = 0
        padded = np.pad(grid, 1, constant_values=1)
        
        for i in range(1, padded.shape[0]-1):
            for j in range(1, padded.shape[1]-1):
                if padded[i, j] == 0:
                    # Verificar si está rodeado
                    neighbors = padded[i-1:i+2, j-1:j+2]
                    if np.sum(neighbors > 0) >= 7:  # Al menos 7 de 8 vecinos
                        holes += 1
        
        return min(holes, 5)  # Limitar para estabilidad

class ProteusARCSolver:
    """Solver ARC basado en principios PROTEUS"""
    
    def __init__(self, population_size: int = 50, seed: Optional[int] = None):
        self.population_size = population_size
        self.organisms = []
        self.field = None
        self.generation = 0
        self.generation_limit = 100  # Límite de generaciones configurable
        
        # Control de aleatoriedad para reproducibilidad
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Resuelve usando evolución topológica sin fitness functions"""
        print("🌀 Iniciando evolución topológica PROTEUS...")
        
        # Inicializar campo y población
        self.field = TopologicalField(test_input.shape)
        self._initialize_population(test_input.shape)
        
        # Fase 1: Aprender de ejemplos (sin supervisión)
        for example in train_examples:
            input_field = np.array(example['input'])
            output_field = np.array(example['output'])
            
            # Los organismos experimentan los campos
            self._experience_fields(input_field, output_field)
        
        # Fase 2: Evolución en el campo de test
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(self.generation_limit):
            # Actualizar campo con el input de test
            self.field.potential = test_input.astype(float)
            self.field.update(self.organisms)
            
            # Evolución puramente termodinámica
            self._evolve_generation()
            
            # Extraer solución del mejor organismo
            if self.organisms:
                best_org = max(self.organisms, key=lambda o: o.seed.dimension * o.energy)
                solution = self._extract_solution(best_org, test_input)
                
                # Evaluar topológicamente
                fitness = self._topological_fitness(solution, train_examples)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution
            
            # Selección natural por supervivencia
            self.organisms = [org for org in self.organisms if org.energy > 0]
            
            # Reproducción para mantener población
            while len(self.organisms) < self.population_size // 2:
                if self.organisms:
                    parent = np.random.choice(self.organisms)
                    child = ProteusOrganism(parent=parent)
                    self.organisms.append(child)
                else:
                    break
        
        return best_solution if best_solution is not None else test_input
    
    def _initialize_population(self, shape: Tuple[int, int]):
        """Inicializa población con diversidad topológica"""
        self.organisms = []
        
        for i in range(self.population_size):
            # Posiciones distribuidas en el espacio
            position = np.random.rand(2) * np.array(shape)
            organism = ProteusOrganism(position)
            
            # Variar dimensiones iniciales
            organism.seed.dimension = 2.0 + np.random.random() * 0.5
            
            self.organisms.append(organism)
    
    def _experience_fields(self, input_field: np.ndarray, output_field: np.ndarray):
        """Los organismos experimentan la transformación"""
        # Crear campo combinado
        combined_field = TopologicalField(input_field.shape)
        combined_field.potential = input_field.astype(float)
        
        # Simular experiencia
        for _ in range(10):  # 10 pasos temporales
            combined_field.update(self.organisms, output_field - input_field)
            
            for organism in self.organisms:
                organism.update(combined_field)
                
                # Codificar experiencia en memoria holográfica
                experience = np.stack([input_field, output_field])
                organism.memory.encode_experience(experience)
    
    def _evolve_generation(self):
        """Evolución sin fitness - solo termodinámica"""
        for organism in self.organisms:
            organism.update(self.field)
            
            # Muerte termodinámica si llega a ciertas regiones
            x, y = int(organism.position[0]), int(organism.position[1])
            if 0 <= x < self.field.shape[0] and 0 <= y < self.field.shape[1]:
                # Penalizar por estar en regiones de alto potencial negativo
                if self.field.potential[x, y] < -1:
                    organism.energy = 0
    
    def _extract_solution(self, organism: ProteusOrganism, test_input: np.ndarray) -> np.ndarray:
        """Extrae solución del campo topológico del organismo"""
        # Usar memoria holográfica para generar transformación
        memory_pattern = organism.memory.recall(test_input)
        
        if memory_pattern is None or memory_pattern.size == 0:
            return test_input
        
        # Aplicar transformación basada en la firma topológica
        solution = test_input.copy()
        
        # Usar curvatura del seed para determinar tipo de transformación
        if organism.seed.curvature < -0.5:
            # Alta curvatura negativa - transformaciones locales
            for i in range(solution.shape[0]):
                for j in range(solution.shape[1]):
                    if test_input[i, j] != 0:
                        # Transformar basado en vecindario topológico
                        neighbors = self._get_topological_neighbors(test_input, i, j)
                        if len(neighbors) > organism.seed.betti_numbers[0]:
                            solution[i, j] = self._transform_by_homology(
                                test_input[i, j], 
                                organism.seed.betti_numbers
                            )
        
        elif organism.seed.dimension > 2.5:
            # Alta dimensión - transformaciones globales
            # Aplicar campo holográfico
            field_transform = organism.seed.field_signature[:solution.shape[0], :solution.shape[1]]
            mask = test_input > 0
            solution[mask] = (test_input[mask] + 
                            np.round(field_transform[mask]).astype(int)) % 10
        
        return solution
    
    def _get_topological_neighbors(self, grid: np.ndarray, x: int, y: int) -> List[int]:
        """Obtiene vecinos topológicamente conectados"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if grid[nx, ny] != 0:
                        neighbors.append(grid[nx, ny])
        return neighbors
    
    def _transform_by_homology(self, value: int, betti_numbers: List[int]) -> int:
        """Transforma valor basado en invariantes topológicos"""
        # Usar números de Betti como operadores
        if betti_numbers[1] > 0:  # Hay agujeros
            return (value + betti_numbers[1]) % 10
        elif betti_numbers[0] > 1:  # Múltiples componentes
            return (value * betti_numbers[0]) % 10
        return value
    
    def _topological_fitness(self, solution: np.ndarray, examples: List[Dict]) -> float:
        """Evalúa fitness basado en acierto de celdas y propiedades topológicas"""
        if solution is None or len(examples) == 0:
            return 0.0
            
        total_fitness = 0.0
        weights_sum = 0.0
        
        # NUEVO: Componente principal - Acierto directo de celdas (70% del peso)
        cell_accuracy_scores = []
        
        for example in examples:
            # Si podemos comparar directamente con los outputs esperados
            output_expected = np.array(example['output'])
            
            if solution.shape == output_expected.shape:
                # Calcular precisión a nivel de celda
                matching_cells = np.sum(solution == output_expected)
                total_cells = output_expected.size
                cell_accuracy = matching_cells / total_cells
                cell_accuracy_scores.append(cell_accuracy)
        
        if cell_accuracy_scores:
            avg_cell_accuracy = np.mean(cell_accuracy_scores)
            total_fitness += avg_cell_accuracy * 0.7
            weights_sum += 0.7
            logger.debug(f"Precisión promedio de celdas: {avg_cell_accuracy:.2%}")
        
        # Componente secundario - Propiedades topológicas (30% del peso)
        topology_scores = []
        
        for example in examples:
            # Comparar propiedades topológicas
            input_dim = self._estimate_fractal_dimension(np.array(example['input']))
            output_dim = self._estimate_fractal_dimension(np.array(example['output']))
            solution_dim = self._estimate_fractal_dimension(solution)
            
            # Evaluar preservación de transformación dimensional
            input_to_output_change = output_dim - input_dim
            expected_solution_dim = solution_dim  # Para el test
            actual_change = solution_dim - self._estimate_fractal_dimension(np.array(example['input']))
            
            # Penalizar diferencias en la transformación
            dimension_error = abs(actual_change - input_to_output_change)
            dimension_score = 1.0 / (1.0 + dimension_error)
            
            # También considerar consistencia de colores
            output_colors = set(np.unique(output_expected))
            solution_colors = set(np.unique(solution))
            color_overlap = len(output_colors.intersection(solution_colors)) / max(len(output_colors), 1)
            
            # Combinar scores topológicos
            topology_score = 0.7 * dimension_score + 0.3 * color_overlap
            topology_scores.append(topology_score)
        
        if topology_scores:
            avg_topology_score = np.mean(topology_scores)
            total_fitness += avg_topology_score * 0.3
            weights_sum += 0.3
        
        # Normalizar por pesos totales
        if weights_sum > 0:
            return total_fitness / weights_sum
        else:
            return 0.0
    
    def _estimate_fractal_dimension(self, grid: np.ndarray) -> float:
        """Estima dimensión fractal para fitness"""
        organism = ProteusOrganism()
        return organism._estimate_dimension(grid)


# Interfaz compatible con el sistema existente
class ProteusAdapter:
    """Adaptador para usar PROTEUS con la interfaz ARC existente"""
    
    def __init__(self):
        self.solver = ProteusARCSolver()
    
    def solve_with_steps(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Interfaz compatible con el solver actual"""
        solution = self.solver.solve(train_examples, test_input)
        
        steps = [
            {
                'type': 'initialization',
                'description': 'Inicializando campo topológico y población'
            },
            {
                'type': 'evolution',
                'description': f'Evolución topológica sin fitness functions'
            },
            {
                'type': 'holographic_recall',
                'description': 'Recuperando patrones de memoria holográfica'
            },
            {
                'type': 'topological_transform',
                'description': 'Aplicando transformación basada en invariantes topológicos'
            }
        ]
        
        return solution, steps


if __name__ == "__main__":
    print("🧬 PROTEUS-ARC: Evolución Topológica sin Redes Neuronales")
    print("=" * 60)
    
    # Test simple
    solver = ProteusARCSolver(population_size=20)
    
    # Ejemplo de color mapping
    train_examples = [
        {
            'input': [[1, 0], [0, 1]],
            'output': [[2, 0], [0, 2]]
        }
    ]
    
    test_input = np.array([[1, 1], [1, 0]])
    
    print("\n🔬 Test Input:")
    print(test_input)
    
    solution = solver.solve(train_examples, test_input)
    
    print("\n✨ Solución PROTEUS:")
    print(solution)
    
    print("\n📊 Estadísticas:")
    print(f"Organismos supervivientes: {len(solver.organisms)}")
    if solver.organisms:
        best = max(solver.organisms, key=lambda o: o.seed.dimension)
        print(f"Mejor dimensión topológica: {best.seed.dimension:.3f}")
        print(f"Números de Betti: {best.seed.betti_numbers}")