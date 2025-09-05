import random
import time
import multiprocessing
from multiprocessing import Process, Value, Array

# Configuración del programa (valores por defecto)
DEFAULT_MATRIX_SIZE = 1000
DEFAULT_NUM_PROCESSES = 4

def calculate_rows_worker(start_row, end_row, A, B, result_flat, rows_A, cols_A, cols_B):
    """
    Función worker para calcular filas de la matriz resultado.
    Cada proceso trabaja en un rango específico de filas.
    """
    for i in range(start_row, end_row):
        for j in range(cols_B):
            # Calcular el producto punto para la posición (i, j)
            dot_product = 0
            for k in range(cols_A):
                dot_product += A[i][k] * B[k][j]
            
            # Asignar el resultado en la matriz plana
            result_flat[i * cols_B + j] = dot_product

def parallel_matrix_multiplication(A, B, num_processes=None):
    """
    Realiza la multiplicación de dos matrices de forma paralela utilizando multiprocessing.
    Divide el trabajo por filas de la matriz resultante.
    
    Args:
        A: Primera matriz (m x n)
        B: Segunda matriz (n x p)
        num_processes: Número de procesos a utilizar. Si es None, usa el número de CPUs disponibles.
    
    Returns:
        Matriz resultado C (m x p)
    """
    # Dimensiones de las matrices
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    
    # Usar el número de CPUs disponibles si no se especifica num_processes
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), rows_A)
    
    # Crear matriz resultado como Array compartido para multiprocessing
    result_flat = Array('d', [0] * (rows_A * cols_B))
    
    # Dividir el trabajo entre procesos
    rows_per_process = rows_A // num_processes
    remaining_rows = rows_A % num_processes
    
    processes = []
    start_row = 0
    
    for process_id in range(num_processes):
        # Calcular cuántas filas procesará este proceso
        process_rows = rows_per_process
        if process_id < remaining_rows:
            process_rows += 1
        
        end_row = start_row + process_rows
        
        # Crear e iniciar el proceso
        process = Process(target=calculate_rows_worker, 
                        args=(start_row, end_row, A, B, result_flat, rows_A, cols_A, cols_B))
        processes.append(process)
        process.start()
        
        start_row = end_row
    
    # Esperar a que todos los procesos terminen
    for process in processes:
        process.join()
    
    # Convertir el array plano de vuelta a matriz 2D
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = result_flat[i * cols_B + j]
    
    return C

def generate_random_matrix(rows, cols):
    """Genera una matriz con valores flotantes aleatorios entre 0 y 1."""
    matrix = [[random.random() for _ in range(cols)] for _ in range(rows)]
    return matrix

if __name__ == "__main__":
    import sys
    
    # Configurar el método de inicio para multiprocessing en Windows
    multiprocessing.set_start_method('spawn', force=True)
    
    # Obtener el tamaño de matriz desde argumentos de línea de comandos
    if len(sys.argv) > 1:
        try:
            MATRIX_SIZE = int(sys.argv[1])
        except ValueError:
            print("Error: El tamaño de matriz debe ser un número entero.")
            MATRIX_SIZE = DEFAULT_MATRIX_SIZE
    else:
        MATRIX_SIZE = DEFAULT_MATRIX_SIZE
    
    # Obtener el número de procesos desde argumentos de línea de comandos
    if len(sys.argv) > 2:
        try:
            NUM_PROCESSES = int(sys.argv[2])
        except ValueError:
            print("Error: El número de procesos debe ser un número entero.")
            NUM_PROCESSES = DEFAULT_NUM_PROCESSES
    else:
        NUM_PROCESSES = DEFAULT_NUM_PROCESSES
    
    print(f"Usando {NUM_PROCESSES} procesos para la multiplicación paralela.")
    print(f"Generando matrices aleatorias de {MATRIX_SIZE}x{MATRIX_SIZE}...")
    
    # Generar las dos matrices a multiplicar
    matrix_A = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)
    matrix_B = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)
    
    print("Matrices generadas. Iniciando multiplicación paralela por procesos...")
    
    # Medir el tiempo de ejecución
    start_time = time.time()
    result_matrix = parallel_matrix_multiplication(matrix_A, matrix_B, NUM_PROCESSES)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print(f"La multiplicación paralela por procesos ha finalizado.")
    print(f"Tiempo total de ejecución: {elapsed_time:.4f} segundos.")
    print(f"Número de procesos utilizados: {NUM_PROCESSES}")
    print(f"Número de CPUs disponibles: {multiprocessing.cpu_count()}")
