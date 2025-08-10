/**
 * Test independiente para verificar que ARCSolver no tiene datos hardcodeados
 */

import { ARCSolver } from '../simulation/ARCSolver';

// Test 1: Color mapping con colores diferentes
function testColorMapping() {
  console.log('\n=== TEST COLOR MAPPING ===');
  const solver = new ARCSolver();
  
  // Ejemplo con mapeo 2->7 (diferente al hardcodeado 1->4)
  const ejemplos = [
    {
      input: [[2, 0, 2], [0, 2, 0], [2, 2, 0]],
      output: [[7, 0, 7], [0, 7, 0], [7, 7, 0]]
    }
  ];
  
  solver.aprender(ejemplos);
  
  // Test con nuevo input
  const test = [[0, 2, 2], [2, 0, 2]];
  const esperado = [[0, 7, 7], [7, 0, 7]];
  const resultado = solver.resolver(test);
  
  console.log('Input test:', test);
  console.log('Output esperado:', esperado);
  console.log('Output obtenido:', resultado);
  console.log('¿Correcto?', JSON.stringify(resultado) === JSON.stringify(esperado) ? '✅' : '❌');
}

// Test 2: Conteo con números diferentes
function testConteo() {
  console.log('\n=== TEST CONTEO ===');
  const solver = new ARCSolver();
  
  const ejemplos = [
    {
      input: [[0, 9, 0], [9, 0, 9], [0, 9, 9]], // 5 nueves
      output: [[5]]
    }
  ];
  
  solver.aprender(ejemplos);
  
  const test = [[9, 9, 0], [0, 0, 9]]; // 3 nueves
  const esperado = [[3]];
  const resultado = solver.resolver(test);
  
  console.log('Input test:', test);
  console.log('Output esperado:', esperado);
  console.log('Output obtenido:', resultado);
  console.log('¿Correcto?', JSON.stringify(resultado) === JSON.stringify(esperado) ? '✅' : '❌');
}

// Test 3: Reflexión con patrón complejo
function testReflexion() {
  console.log('\n=== TEST REFLEXIÓN ===');
  const solver = new ARCSolver();
  
  const ejemplos = [
    {
      input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
      output: [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    }
  ];
  
  solver.aprender(ejemplos);
  
  const test = [[5, 0, 8], [0, 3, 0]];
  const esperado = [[8, 0, 5], [0, 3, 0]];
  const resultado = solver.resolver(test);
  
  console.log('Input test:', test);
  console.log('Output esperado:', esperado);
  console.log('Output obtenido:', resultado);
  console.log('¿Correcto?', JSON.stringify(resultado) === JSON.stringify(esperado) ? '✅' : '❌');
}

// Test 4: Simetría
function testSimetria() {
  console.log('\n=== TEST SIMETRÍA ===');
  const solver = new ARCSolver();
  
  const ejemplos = [
    {
      input: [[1, 0, 1], [0, 1, 0], [1, 0, 1]], // Simétrico
      output: [[1]]
    },
    {
      input: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], // No simétrico
      output: [[0]]
    }
  ];
  
  solver.aprender(ejemplos);
  
  const test = [[2, 3, 2], [3, 4, 3], [2, 3, 2]]; // Simétrico
  const esperado = [[1]];
  const resultado = solver.resolver(test);
  
  console.log('Input test:', test);
  console.log('Output esperado:', esperado);
  console.log('Output obtenido:', resultado);
  console.log('¿Correcto?', JSON.stringify(resultado) === JSON.stringify(esperado) ? '✅' : '❌');
}

// Ejecutar todos los tests
export function runTests() {
  console.log('🔍 VERIFICACIÓN DE INTEGRIDAD - NO HAY DATOS FAKE');
  console.log('================================================');
  
  testColorMapping();
  testConteo();
  testReflexion();
  testSimetria();
  
  console.log('\n✨ Tests completados. Si todos pasan, el sistema NO tiene datos hardcodeados.');
}

// Para ejecutar desde la consola del navegador:
// import('./test/TestARCSolver.js').then(m => m.runTests());