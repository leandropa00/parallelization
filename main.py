#!/usr/bin/env python3
"""
Selector principal para diferentes métodos de paralelización de multiplicación de matrices.
Permite elegir entre:
1. Sequential (secuencial)
2. Threads (hilos)
3. Multiprocessing (múltiples procesos)
4. MPI (Message Passing Interface)
"""

import subprocess
import sys
import os

# Configuración del programa (valores por defecto)
DEFAULT_MATRIX_SIZE = 1000
DEFAULT_NUM_THREADS = 4

# Variables globales que se pueden modificar
MATRIX_SIZE = DEFAULT_MATRIX_SIZE
NUM_THREADS = DEFAULT_NUM_THREADS

def run_sequential():
    """Ejecuta el archivo sequential.py directamente."""
    print("=" * 60)
    print("EJECUTANDO MULTIPLICACIÓN SECUENCIAL")
    print("=" * 60)
    print(f"Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    
    try:
        # Pasar las variables como argumentos de línea de comandos
        result = subprocess.run([sys.executable, "sequential.py", str(MATRIX_SIZE)], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Error ejecutando sequential.py: {e}")
        return False

def run_threads():
    """Ejecuta el archivo threads.py directamente."""
    print("=" * 60)
    print("EJECUTANDO MULTIPLICACIÓN CON HILOS")
    print("=" * 60)
    print(f"Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Número de hilos: {NUM_THREADS}")
    
    try:
        # Pasar las variables como argumentos de línea de comandos
        result = subprocess.run([sys.executable, "threads.py", str(MATRIX_SIZE), str(NUM_THREADS)], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Error ejecutando threads.py: {e}")
        return False

def run_multiprocess():
    """Ejecuta el archivo multiprocess.py directamente."""
    print("=" * 60)
    print("EJECUTANDO MULTIPLICACIÓN CON MULTIPROCESSING")
    print("=" * 60)
    print(f"Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Número de procesos: {NUM_THREADS}")
    
    try:
        # Pasar las variables como argumentos de línea de comandos
        result = subprocess.run([sys.executable, "multiprocess.py", str(MATRIX_SIZE), str(NUM_THREADS)], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Error ejecutando multiprocess.py: {e}")
        return False

def run_mpi():
    """Ejecuta el archivo mpi.py directamente con mpirun."""
    print("=" * 60)
    print("EJECUTANDO MULTIPLICACIÓN CON MPI")
    print("=" * 60)
    print(f"Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Número de procesos MPI: {NUM_THREADS}")
    
    try:
        # Intentar ejecutar con mpirun
        result = subprocess.run(["mpirun", "-n", str(NUM_THREADS), sys.executable, "mpi.py", str(MATRIX_SIZE)], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except FileNotFoundError:
        print("mpirun no encontrado. Intentando ejecutar mpi.py directamente...")
        try:
            result = subprocess.run([sys.executable, "mpi.py", str(MATRIX_SIZE)], 
                                  capture_output=False, 
                                  text=True, 
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
            return result.returncode == 0
        except Exception as e:
            print(f"Error ejecutando mpi.py: {e}")
            return False
    except Exception as e:
        print(f"Error ejecutando MPI: {e}")
        return False

def configure_settings():
    """Permite configurar las variables desde la terminal."""
    global MATRIX_SIZE, NUM_THREADS
    
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN DE PARÁMETROS")
    print("=" * 60)
    print(f"Configuración actual:")
    print(f"- Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"- Número de hilos/procesos: {NUM_THREADS}")
    print("=" * 60)
    
    while True:
        try:
            print("\nOpciones de configuración:")
            print("1. Cambiar tamaño de matriz")
            print("2. Cambiar número de hilos/procesos")
            print("3. Restaurar valores por defecto")
            print("0. Volver al menú principal")
            
            choice = input("\nSelecciona una opción (0-3): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                new_size = input(f"Ingresa el nuevo tamaño de matriz (actual: {MATRIX_SIZE}): ").strip()
                if new_size.isdigit():
                    MATRIX_SIZE = int(new_size)
                    print(f"✓ Tamaño de matriz actualizado a: {MATRIX_SIZE}x{MATRIX_SIZE}")
                else:
                    print("✗ Error: Debe ser un número entero positivo")
            elif choice == '2':
                new_threads = input(f"Ingresa el nuevo número de hilos/procesos (actual: {NUM_THREADS}): ").strip()
                if new_threads.isdigit() and int(new_threads) > 0:
                    NUM_THREADS = int(new_threads)
                    print(f"✓ Número de hilos/procesos actualizado a: {NUM_THREADS}")
                else:
                    print("✗ Error: Debe ser un número entero positivo")
            elif choice == '3':
                MATRIX_SIZE = DEFAULT_MATRIX_SIZE
                NUM_THREADS = DEFAULT_NUM_THREADS
                print(f"✓ Valores restaurados a los por defecto:")
                print(f"  - Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
                print(f"  - Número de hilos/procesos: {NUM_THREADS}")
            else:
                print("✗ Opción inválida. Por favor, selecciona un número del 0 al 3.")
                
        except KeyboardInterrupt:
            print("\n\nVolviendo al menú principal...")
            break
        except Exception as e:
            print(f"✗ Error inesperado: {e}")

def show_menu():
    """Muestra el menú de opciones."""
    print("\n" + "=" * 60)
    print("SELECTOR DE MÉTODOS DE PARALELIZACIÓN")
    print("Multiplicación de Matrices")
    print("=" * 60)
    print("1. Sequential (Secuencial)")
    print("2. Threads (Hilos)")
    print("3. Multiprocessing (Múltiples procesos)")
    print("4. MPI (Message Passing Interface)")
    print("5. Configurar parámetros")
    print("0. Salir")
    print("=" * 60)

def main():
    """Función principal del selector."""
    print("Bienvenido al selector de métodos de paralelización para multiplicación de matrices")
    print(f"Configuración actual:")
    print(f"- Tamaño de matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"- Número de hilos/procesos: {NUM_THREADS}")
    
    while True:
        show_menu()
        
        try:
            choice = input("\nSelecciona una opción (0-5): ").strip()
            
            if choice == '0':
                print("¡Hasta luego!")
                break
            elif choice == '1':
                success = run_sequential()
                if not success:
                    print("Error ejecutando sequential.py")
            elif choice == '2':
                success = run_threads()
                if not success:
                    print("Error ejecutando threads.py")
            elif choice == '3':
                success = run_multiprocess()
                if not success:
                    print("Error ejecutando multiprocess.py")
            elif choice == '4':
                success = run_mpi()
                if not success:
                    print("Error ejecutando mpi.py")
            elif choice == '5':
                configure_settings()
            else:
                print("Opción inválida. Por favor, selecciona un número del 0 al 5.")
                
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")
            print("Por favor, intenta de nuevo.")
        
        input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()
