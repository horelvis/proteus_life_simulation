"""
Modelos Pydantic para la API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum


class OrganType(str, Enum):
    PHOTOSENSOR = "photosensor"
    CHEMORECEPTOR = "chemoreceptor"
    FLAGELLUM = "flagellum"
    CILIA = "cilia"
    MEMBRANE = "membrane"
    VACUOLE = "vacuole"
    PSEUDOPOD = "pseudopod"
    CRYSTALLIN = "crystallin"
    PIGMENT_SPOT = "pigment_spot"
    NERVE_NET = "nerve_net"


class OrganData(BaseModel):
    type: OrganType
    expression: float = Field(ge=0, le=1)
    development_stage: float = Field(ge=0, le=1)
    functionality: float = Field(ge=0, le=1)
    cost: float = Field(ge=0)


class CapabilitiesData(BaseModel):
    vision: float = Field(ge=0, le=1)
    chemotaxis: float = Field(ge=0, le=1)
    motility: float = Field(ge=0)
    protection: float = Field(ge=0, le=1)
    efficiency: float = Field(ge=0)


class OrganismData(BaseModel):
    id: str
    type: str = "protozoa"
    position: Dict[str, float]  # {"x": float, "y": float}
    velocity: Dict[str, float]
    generation: int
    age: float
    energy: float
    alive: bool
    phenotype: str
    organs: List[OrganData]
    capabilities: CapabilitiesData
    trajectory: Optional[List[Dict[str, float]]] = None
    color: str = "#00CED1"


class PredatorData(BaseModel):
    id: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    is_attacking: bool
    light_radius: float
    attack_cooldown: int


class FieldData(BaseModel):
    """Datos del campo para visualización"""
    light_field: Optional[List[List[float]]] = None  # Matriz 2D simplificada
    nutrient_field: Optional[List[List[float]]] = None
    field_resolution: int = 50  # Resolución reducida para web


class SimulationConfig(BaseModel):
    world_size: Tuple[int, int] = (800, 600)
    initial_organisms: int = 50
    initial_predators: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    enable_organs: bool = True
    physics_params: Dict[str, float] = {
        "viscosity": 0.8,
        "temperature": 20.0,
        "light_decay": 0.95,
        "nutrient_regeneration": 0.001
    }


class SimulationState(BaseModel):
    id: str
    time: float
    generation: int
    status: str
    organisms: List[OrganismData]
    predators: List[PredatorData]
    field_data: Optional[FieldData] = None
    statistics: Dict[str, float]


class WorldUpdate(BaseModel):
    """Actualización incremental del mundo"""
    time: float
    organisms: List[OrganismData]
    predators: List[PredatorData]
    nutrients: List[Dict] = []  # Nutrientes activos
    field_patches: Optional[Dict] = None  # Solo las zonas que cambiaron
    events: List[Dict]  # Eventos importantes (nacimientos, muertes, etc.)
    statistics_delta: Dict[str, float]


class SimulationEvent(BaseModel):
    type: str  # "birth", "death", "mutation", "attack", etc.
    time: float
    organism_id: Optional[str] = None
    data: Dict