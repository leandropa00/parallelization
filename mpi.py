import random
import time
import numpy as np

# Configuración del programa (valor por defecto)
DEFAULT_MATRIX_SIZE = 1000

def mpi_matrix_multiplication(A, B):
    """
    Realiza la multiplicación de dos matrices usando MPI.
    
    Args:
        A: Primera matriz (m x n)
        B: Segunda matriz (n x p)
    
    Returns:
        Matriz resultado C (m x p)
    """
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("mpi4py no está instalado. Instálalo con: pip install mpi4py")
    
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Solo el proceso 0 imprime información
    if rank == 0:
        print(f"Usando {size} procesos MPI")
        print(f"Tamaño de matriz: {len(A)}x{len(A[0])}")
    
    # Dimensiones de las matrices
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    
    # Convertir a arrays de NumPy para mejor rendimiento
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    
    # Distribuir las filas de A entre los procesos
    rows_per_process = rows_A // size
    remaining_rows = rows_A % size
    
    # Calcular el rango de filas para este proceso
    if rank < remaining_rows:
        start_row = rank * (rows_per_process + 1)
        end_row = start_row + rows_per_process + 1
    else:
        start_row = rank * rows_per_process + remaining_rows
        end_row = start_row + rows_per_process
    
    # Cada proceso calcula su parte de la matriz resultado
    local_result = np.zeros((end_row - start_row, cols_B), dtype=np.float64)
    
    # Multiplicación local
    for i in range(start_row, end_row):
        for j in range(cols_B):
            dot_product = 0
            for k in range(cols_A):
                dot_product += A_np[i][k] * B_np[k][j]
            local_result[i - start_row][j] = dot_product
    
    # Recopilar resultados en el proceso 0
    if rank == 0:
        # Inicializar matriz resultado completa
        result_matrix = np.zeros((rows_A, cols_B), dtype=np.float64)
        
        # Copiar resultado local del proceso 0
        result_matrix[start_row:end_row] = local_result
        
        # Recibir resultados de otros procesos
        for source_rank in range(1, size):
            # Calcular rango de filas del proceso fuente
            if source_rank < remaining_rows:
                source_start = source_rank * (rows_per_process + 1)
                source_end = source_start + rows_per_process + 1
            else:
                source_start = source_rank * rows_per_process + remaining_rows
                source_end = source_start + rows_per_process
            
            # Recibir datos del proceso fuente
            source_data = np.zeros((source_end - source_start, cols_B), dtype=np.float64)
            comm.Recv(source_data, source=source_rank, tag=0)
            
            # Copiar al resultado final
            result_matrix[source_start:source_end] = source_data
        
        return result_matrix.tolist()
    
    else:
        # Enviar resultado local al proceso 0
        comm.Send(local_result, dest=0, tag=0)
        return None

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
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            print(f"Generando matrices aleatorias de {MATRIX_SIZE}x{MATRIX_SIZE}...")
            print(f"Usando {size} procesos MPI.")
        
        # Generar matrices solo en el proceso 0
        if rank == 0:
            matrix_A = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)
            matrix_B = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)
        else:
            matrix_A = None
            matrix_B = None
        
        # Broadcast de las matrices a todos los procesos
        matrix_A = comm.bcast(matrix_A, root=0)
        matrix_B = comm.bcast(matrix_B, root=0)
        
        if rank == 0:
            print("Matrices generadas. Iniciando multiplicación MPI...")
        
        # Medir el tiempo de ejecución
        start_time = time.time()
        result_matrix = mpi_matrix_multiplication(matrix_A, matrix_B)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if rank == 0:
            print(f"La multiplicación MPI ha finalizado.")
            print(f"Tiempo total de ejecución: {elapsed_time:.4f} segundos.")
    
    except ImportError:
        print("Error: mpi4py no está instalado.")
        print("Para instalar mpi4py:")
        print("1. Instala MPI: sudo apt-get install libopenmpi-dev (Ubuntu/Debian)")
        print("2. Instala mpi4py: pip install mpi4py")
