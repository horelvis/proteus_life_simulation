#!/usr/bin/env python3
"""
ARC Logical Reasoning Network Server
Sistema puro de razonamiento lógico sin simulación de vida
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from arc import ARCSolver

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class ARCExample(BaseModel):
    input: List[List[int]]
    output: List[List[int]]

class ARCPuzzle(BaseModel):
    train: List[ARCExample]
    test: List[Dict[str, List[List[int]]]]

class SolutionResponse(BaseModel):
    solution: List[List[int]]
    reasoning: Dict[str, Any]
    confidence: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🧠 ARC Logical Reasoning Network starting...")
    yield
    # Shutdown
    logger.info("Shutting down reasoning network...")

app = FastAPI(
    title="ARC Reasoning API",
    description="Logical Reasoning Network for ARC puzzles",
    version="6.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "ARC Logical Reasoning Network",
        "version": "6.0.0",
        "architecture": "3-layer (Macro→Meso→Micro)",
        "status": "operational"
    }

@app.post("/solve", response_model=SolutionResponse)
async def solve_puzzle(puzzle: ARCPuzzle):
    """
    Resuelve un puzzle ARC usando razonamiento lógico
    """
    try:
        # Convertir a formato numpy
        train_examples = [
            {
                'input': np.array(ex.input).tolist(),
                'output': np.array(ex.output).tolist()
            }
            for ex in puzzle.train
        ]
        
        test_input = np.array(puzzle.test[0]['input'])
        
        # Resolver con red de razonamiento
        solver = ARCSolver()
        solution = solver.reason(train_examples, test_input)
        
        # Extraer cadena de razonamiento
        reasoning = {
            "macro_inferences": [
                {
                    "premise": inf.premise,
                    "conclusion": inf.conclusion,
                    "confidence": inf.confidence
                }
                for inf in solver.inferences[:3]  # Top 3 inferencias
            ],
            "patterns_detected": len(solver.logical_patterns),
            "reasoning_steps": 3  # Macro → Meso → Micro
        }
        
        # Calcular confianza
        confidence = np.mean([inf.confidence for inf in solver.inferences]) if solver.inferences else 0.5
        
        return SolutionResponse(
            solution=solution.tolist(),
            reasoning=reasoning,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Error solving puzzle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "reasoning_engine": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)