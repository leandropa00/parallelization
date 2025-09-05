import random
import time
import threading

# Configuración del programa (valores por defecto)
DEFAULT_MATRIX_SIZE = 1000
DEFAULT_NUM_THREADS = 12

def parallel_matrix_multiplication(A, B, num_threads=None):
    """
    Realiza la multiplicación de dos matrices de forma paralela utilizando threading.
    Divide el trabajo por filas de la matriz resultante.
    
    Args:
        A: Primera matriz (m x n)
        B: Segunda matriz (n x p)
        num_threads: Número de hilos a utilizar. Si es None, usa el número de CPUs disponibles.
    
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
    
    # Usar el número de CPUs disponibles si no se especifica num_threads
    if num_threads is None:
        num_threads = min(threading.active_count(), rows_A)
    
    # Inicializar la matriz resultado con ceros
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    def calculate_rows(start_row, end_row):
        """
        Calcula las filas de la matriz resultado desde start_row hasta end_row (exclusivo).
        Cada hilo trabaja en un rango específico de filas.
        """
        for i in range(start_row, end_row):
            for j in range(cols_B):
                # Calcular el producto punto para la posición (i, j)
                dot_product = 0
                for k in range(cols_A):
                    dot_product += A[i][k] * B[k][j]
                
                # Asignar el resultado (no necesitamos lock ya que cada hilo escribe en filas únicas)
                C[i][j] = dot_product
    
    # Dividir el trabajo entre hilos
    rows_per_thread = rows_A // num_threads
    remaining_rows = rows_A % num_threads
    
    threads = []
    start_row = 0
    
    for thread_id in range(num_threads):
        # Calcular cuántas filas procesará este hilo
        thread_rows = rows_per_thread
        if thread_id < remaining_rows:
            thread_rows += 1
        
        end_row = start_row + thread_rows
        
        # Crear e iniciar el hilo
        thread = threading.Thread(target=calculate_rows, args=(start_row, end_row))
        threads.append(thread)
        thread.start()
        
        start_row = end_row
    
    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()
    
    return C

def generate_random_matrix(rows, cols):
    """Genera una matriz con valores flotantes aleatorios entre 0 y 1."""
    matrix = [[random.random() for _ in range(cols)] for _ in range(rows)]
    return matrix

if __name__ == "__main__":
    import sys
    
    # Obtener el tamaño de matriz desde argumentos de línea de comandos
    if len(sys.argv) > 1:
        try:
            MATRIX_SIZE = int(sys.argv[1])
        except ValueError:
            print("Error: El tamaño de matriz debe ser un número entero.")
            MATRIX_SIZE = DEFAULT_MATRIX_SIZE
    else:
        MATRIX_SIZE = DEFAULT_MATRIX_SIZE
    
    # Pueden ajustar este valor si su máquina tiene más o menos recursos.
    # ¡Cuidado con valores muy grandes que puedan colgar su sistema!
    
    print(f"Generando matrices aleatorias de {MATRIX_SIZE}x{MATRIX_SIZE}...")
    
    # Generar las dos matrices a multiplicar
    matrix_A = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)
    matrix_B = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)
    
    print("Matrices generadas. Iniciando multiplicación paralela por hilos...")
    
    # Medir el tiempo de ejecución
    start_time = time.time()
    result_matrix = parallel_matrix_multiplication(matrix_A, matrix_B)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print(f"La multiplicación paralela por hilos ha finalizado.")
    print(f"Tiempo total de ejecución: {elapsed_time:.4f} segundos.")
