from population import Population
from selection import elitist_selection, roulette_selection, ranking_selection_v1, ranking_selection_v2, tournament_selection, steady_state_selection, crowding_selection
from crossover import uniform_crossover, arithmetic_crossover, flat_crossover, blx_alpha_crossover, pmx_crossover
from mutation import random_mutation, uniform_mutation, non_uniform_mutation, swap_mutation
from visualization import plot_evolution
import csv
from datetime import datetime

import numpy as np
import sys
import tkinter as tk
from tkinter import simpledialog
import copy
import re
import math


# Hyperparameters configuration window
def get_hyperparameters():
    import tkinter as tk
    from tkinter import ttk
    
    root = tk.Tk()
    root.title("🧬 Configuración del Algoritmo Genético")
    
    # Make window resizable and set better initial size
    root.geometry("540x600+400+150")
    root.minsize(500, 400)
    root.configure(bg='#f0f0f0')
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
    style.configure('Heading.TLabel', font=('Arial', 10, 'bold'))
    style.configure('Config.TLabel', font=('Arial', 9), foreground='#333333')
    style.configure('Custom.TFrame', background='#ffffff', relief='raised', borderwidth=1)
    style.configure('Submit.TButton', font=('Arial', 10, 'bold'), padding=(20, 8))
    
    # Create main container with scrollbar
    canvas = tk.Canvas(root, bg='#f0f0f0', highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    # Configure scrolling
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Main frame with elegant styling inside scrollable area
    main_frame = ttk.Frame(scrollable_frame, style='Custom.TFrame', padding=25)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    # Title
    title_label = ttk.Label(main_frame, text="Configuración de Parámetros", style='Title.TLabel')
    title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
    
    # Create sections
    # Population Parameters
    pop_label = ttk.Label(main_frame, text="📊 Parámetros de Población", style='Heading.TLabel')
    pop_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 10))
    
    ttk.Label(main_frame, text="Tamaño de la población:", style='Config.TLabel').grid(row=2, column=0, sticky='w', pady=6, padx=(10, 0))
    pop_size_var = tk.IntVar(value=20)
    pop_size_entry = ttk.Entry(main_frame, textvariable=pop_size_var, width=15, font=('Arial', 9))
    pop_size_entry.grid(row=2, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Número de alelos:", style='Config.TLabel').grid(row=3, column=0, sticky='w', pady=6, padx=(10, 0))
    dna_length_var = tk.IntVar(value=10)
    dna_length_entry = ttk.Entry(main_frame, textvariable=dna_length_var, width=15, font=('Arial', 9))
    dna_length_entry.grid(row=3, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Número de generaciones:", style='Config.TLabel').grid(row=4, column=0, sticky='w', pady=6, padx=(10, 0))
    generations_var = tk.IntVar(value=50)
    generations_entry = ttk.Entry(main_frame, textvariable=generations_var, width=15, font=('Arial', 9))
    generations_entry.grid(row=4, column=1, pady=6, padx=10, sticky='ew')

    # Genetic Parameters
    genetic_label = ttk.Label(main_frame, text="🧬 Parámetros Genéticos", style='Heading.TLabel')
    genetic_label.grid(row=5, column=0, columnspan=3, sticky='w', pady=(15, 10))
    
    ttk.Label(main_frame, text="Codificación:", style='Config.TLabel').grid(row=6, column=0, sticky='w', pady=6, padx=(10, 0))
    encoding_var = tk.StringVar(value='real')
    encoding_menu = ttk.Combobox(main_frame, textvariable=encoding_var, values=['binary', 'real'], 
                                state='readonly', width=13, font=('Arial', 9))
    encoding_menu.grid(row=6, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Método de selección:", style='Config.TLabel').grid(row=7, column=0, sticky='w', pady=6, padx=(10, 0))
    selection_var = tk.StringVar(value='elitist')
    selection_menu = ttk.Combobox(main_frame, textvariable=selection_var, 
                                 values=['elitist', 'roulette', 'ranking_v1', 'ranking_v2', 'tournament', 'steady_state', 'crowding'], 
                                 state='readonly', width=13, font=('Arial', 9))
    selection_menu.grid(row=7, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Método de cruce:", style='Config.TLabel').grid(row=8, column=0, sticky='w', pady=6, padx=(10, 0))
    crossover_var = tk.StringVar(value='uniform')
    crossover_menu = ttk.Combobox(main_frame, textvariable=crossover_var, 
                                 values=['uniform', 'arithmetic', 'flat', 'blx_alpha', 'pmx'], 
                                 state='readonly', width=13, font=('Arial', 9))
    crossover_menu.grid(row=8, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Probabilidad de cruce (Pc):", style='Config.TLabel').grid(row=9, column=0, sticky='w', pady=6, padx=(10, 0))
    pc_var = tk.DoubleVar(value=0.8)
    pc_entry = ttk.Entry(main_frame, textvariable=pc_var, width=15, font=('Arial', 9))
    pc_entry.grid(row=9, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Método de mutación:", style='Config.TLabel').grid(row=10, column=0, sticky='w', pady=6, padx=(10, 0))
    mutation_var = tk.StringVar(value='random')
    mutation_menu = ttk.Combobox(main_frame, textvariable=mutation_var, 
                                values=['random', 'uniform', 'non_uniform', 'swap'], 
                                state='readonly', width=13, font=('Arial', 9))
    mutation_menu.grid(row=10, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Probabilidad de mutación (Pm):", style='Config.TLabel').grid(row=11, column=0, sticky='w', pady=6, padx=(10, 0))
    pm_var = tk.DoubleVar(value=0.01)
    pm_entry = ttk.Entry(main_frame, textvariable=pm_var, width=15, font=('Arial', 9))
    pm_entry.grid(row=11, column=1, pady=6, padx=10, sticky='ew')

    ttk.Label(main_frame, text="Límites (min,max):", style='Config.TLabel').grid(row=12, column=0, sticky='w', pady=6, padx=(10, 0))
    bounds_var = tk.StringVar(value='0,10')
    bounds_entry = ttk.Entry(main_frame, textvariable=bounds_var, width=15, font=('Arial', 9))
    bounds_entry.grid(row=12, column=1, pady=6, padx=10, sticky='ew')

    # Configure column weights for better stretching
    main_frame.grid_columnconfigure(1, weight=1)
    
    # Button frame to keep it always visible
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=13, column=0, columnspan=3, pady=20)
    
    def submit():
        root.quit()
        root.destroy()

    submit_btn = ttk.Button(button_frame, text="🚀 Iniciar Algoritmo", command=submit, style='Submit.TButton')
    submit_btn.pack()
    
    # Mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    root.mainloop()
    
    try:
        bounds = tuple(float(x.strip()) for x in bounds_var.get().split(','))
    except ValueError:
        print("Error: El formato de Límites (min,max) es incorrecto. Usando valores predeterminados (0, 10).")
        bounds = (0, 10)
    
    return {
        'POP_SIZE': pop_size_var.get(),
        'DNA_LENGTH': dna_length_var.get(),
        'GENERATIONS': generations_var.get(),
        'ENCODING': encoding_var.get(),
        'BOUNDS': bounds,
        'SELECTION_METHOD': selection_var.get(),
        'CROSSOVER_METHOD': crossover_var.get(),
        'PC': pc_var.get(),
        'MUTATION_METHOD': mutation_var.get(),
        'PM': pm_var.get()
    }

# Get hyperparameters from user
params = get_hyperparameters()
POP_SIZE = params['POP_SIZE']
DNA_LENGTH = params['DNA_LENGTH']
GENERATIONS = params['GENERATIONS']
ENCODING = params['ENCODING']
BOUNDS = params['BOUNDS']
PC = params['PC']
PM = params['PM']

# Selection method mapping
selection_methods = {
    'elitist': elitist_selection,
    'roulette': roulette_selection,
    'ranking_v1': ranking_selection_v1,
    'ranking_v2': ranking_selection_v2,
    'tournament': tournament_selection,
    'steady_state': steady_state_selection,
    'crowding': crowding_selection
}
SELECTION_METHOD = selection_methods[params['SELECTION_METHOD']]

# Crossover method mapping
crossover_methods = {
    'uniform': uniform_crossover,
    'arithmetic': arithmetic_crossover,
    'flat': flat_crossover,
    'blx_alpha': blx_alpha_crossover,
    'pmx': pmx_crossover
}
CROSSOVER_METHOD = crossover_methods[params['CROSSOVER_METHOD']]

# Mutation method mapping
mutation_methods = {
    'random': random_mutation,
    'uniform': uniform_mutation,
    'non_uniform': non_uniform_mutation,
    'swap': swap_mutation
}
MUTATION_METHOD = mutation_methods[params['MUTATION_METHOD']]


# MEJORA 1: Conversión de sintaxis mejorada (sin la conversión problemática de exponentes)
def convert_to_python_syntax(expr):
    expr = expr.strip()
    # Remove assignment (e.g., z = ... or f(x, y) = ...)
    expr = re.sub(r'^\s*[a-zA-Z_]\w*\s*(\([^)]*\))?\s*=\s*', '', expr)
    # Replace ^ with ** for exponentiation
    expr = re.sub(r'\^', '**', expr)
    # Replace math functions with numpy
    expr = re.sub(r'cos\s*\(([^)]+)\)', r'np.cos(\1)', expr)
    expr = re.sub(r'sin\s*\(([^)]+)\)', r'np.sin(\1)', expr)
    expr = re.sub(r'tan\s*\(([^)]+)\)', r'np.tan(\1)', expr)
    expr = re.sub(r'log\s*\(([^)]+)\)', r'np.log(\1)', expr)
    expr = re.sub(r'exp\s*\(([^)]+)\)', r'np.exp(\1)', expr)
    expr = re.sub(r'sqrt\s*\(([^)]+)\)', r'np.sqrt(\1)', expr)
    # Replace variables x, y, z, w, ... with x[0], x[1], x[2], x[3], ...
    var_map = ['x', 'y', 'z', 'w', 'u', 'v', 't', 's', 'r', 'q', 'p']
    for idx, var in enumerate(var_map):
        expr = re.sub(rf'\b{var}\b', f'x[{idx}]', expr)
    return expr


# MEJORA 2: Visualización mejorada con soporte para encoding real
def show_all_generations_window(generations_data, encoding):
    import tkinter as tk
    from tkinter import ttk
    import math
    
    root = tk.Tk()
    root.title("🧬 Resultados de Todas las Generaciones")
    
    # Set window size based on screen size and make it resizable
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate optimal window size (80% of screen)
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    
    # Center the window
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.minsize(700, 500)  # Set minimum size
    root.configure(bg='#f5f5f5')
    
    # Create main frame with elegant styling
    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Navigation.TFrame', background='#ffffff', relief='raised', borderwidth=1)
    style.configure('Nav.TButton', padding=(12, 6), font=('Arial', 9))
    style.configure('PageInfo.TLabel', font=('Arial', 10, 'bold'), background='#ffffff')
    style.configure('Content.TFrame', background='#ffffff', relief='sunken', borderwidth=1)
    
    # Variables for pagination
    current_page = tk.IntVar(value=0)
    tabs_per_page = 8  # Show 8 tabs per page for better spacing
    total_generations = len(generations_data)
    total_pages = math.ceil(total_generations / tabs_per_page)
    
    # Create navigation frame
    nav_frame = ttk.Frame(main_frame, style='Navigation.TFrame', padding=10)
    nav_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Navigation controls
    def update_page():
        page = current_page.get()
        
        # Clear existing notebook
        for widget in content_frame.winfo_children():
            widget.destroy()
        
        # Create new notebook for current page
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Calculate range for current page
        start_idx = page * tabs_per_page
        end_idx = min(start_idx + tabs_per_page, total_generations)
        
        # Add tabs for current page
        for gen_idx in range(start_idx, end_idx):
            population = generations_data[gen_idx]
            
            # Create frame for each generation
            frame = ttk.Frame(notebook)
            
            # Determine tab name
            if gen_idx == 0:
                tab_name = "🌱 Inicial"
            elif gen_idx == total_generations - 1:
                tab_name = "🎯 Final"
            else:
                tab_name = f"Gen {gen_idx}"
            
            notebook.add(frame, text=tab_name)
            
            # Create treeview with improved configuration
            tree = ttk.Treeview(frame, columns=["#", "ADN", "Fitness"], show="headings", height=15)
            
            # Create scrollbars
            vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            # Pack scrollbars and treeview
            vsb.pack(side='right', fill='y')
            hsb.pack(side='bottom', fill='x')
            tree.pack(side='left', fill=tk.BOTH, expand=True)
            
            # Configure columns with better proportions
            tree.heading("#", text="No.", anchor=tk.CENTER)
            tree.heading("ADN", text="ADN", anchor=tk.CENTER)
            tree.heading("Fitness", text="Fitness", anchor=tk.CENTER)
            
            # Adjust column widths dynamically
            tree.column("#", anchor=tk.CENTER, width=60, minwidth=50)
            tree.column("ADN", anchor=tk.CENTER, width=220, minwidth=180)
            tree.column("Fitness", anchor=tk.CENTER, width=120, minwidth=100)
            
            # Add data to tree with improved formatting
            for i, ind in enumerate(population.individuals):
                # MEJORA: Mostrar ADN según encoding
                if encoding == 'real':
                    adn_str = ', '.join([f'{val:.4f}' for val in ind.dna])
                else:
                    binary_dna = [1 if bit >= 0.5 else 0 for bit in ind.dna]
                    adn_str = ''.join(str(b) for b in binary_dna)
                
                # Truncate DNA string if too long for better visibility
                if len(adn_str) > 24:
                    adn_display = adn_str[:21] + "..."
                else:
                    adn_display = adn_str
                
                values = [f"{i+1:02d}", adn_display, f"{ind.fitness:.6f}"]
                
                # Add alternating row colors for better readability
                tag = 'oddrow' if i % 2 else 'evenrow'
                tree.insert("", "end", values=values, tags=(tag,))
            
            # Configure row colors with better contrast
            tree.tag_configure('oddrow', background='#f8f9fa')
            tree.tag_configure('evenrow', background='white')
        
        # Update navigation buttons state
        prev_btn.config(state='normal' if page > 0 else 'disabled')
        next_btn.config(state='normal' if page < total_pages - 1 else 'disabled')
        
        # Update page info
        page_info.config(text=f"Página {page + 1} de {total_pages} | Generaciones {start_idx + 1}-{end_idx} de {total_generations}")
    
    def prev_page():
        if current_page.get() > 0:
            current_page.set(current_page.get() - 1)
            update_page()
    
    def next_page():
        if current_page.get() < total_pages - 1:
            current_page.set(current_page.get() + 1)
            update_page()
    
    def first_page():
        current_page.set(0)
        update_page()
    
    def last_page():
        current_page.set(total_pages - 1)
        update_page()
    
    # Navigation buttons with better styling
    ttk.Button(nav_frame, text="⏮ Primera", command=first_page, style='Nav.TButton').pack(side=tk.LEFT, padx=(0, 5))
    prev_btn = ttk.Button(nav_frame, text="◀ Anterior", command=prev_page, style='Nav.TButton')
    prev_btn.pack(side=tk.LEFT, padx=5)
    
    # Page info in the center
    page_info = ttk.Label(nav_frame, text="", style='PageInfo.TLabel')
    page_info.pack(side=tk.LEFT, expand=True)
    
    next_btn = ttk.Button(nav_frame, text="Siguiente ▶", command=next_page, style='Nav.TButton')
    next_btn.pack(side=tk.RIGHT, padx=5)
    ttk.Button(nav_frame, text="Última ⏭", command=last_page, style='Nav.TButton').pack(side=tk.RIGHT, padx=(5, 0))
    
    # Create content frame for the notebook
    content_frame = ttk.Frame(main_frame, style='Content.TFrame', padding=5)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    # Add status bar with enhanced information
    status_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
    status_frame.pack(fill=tk.X)
    
    status_label = ttk.Label(status_frame, 
                           text=f"📊 Total: {total_generations} generaciones | 👥 Población: {len(generations_data[0].individuals) if generations_data else 0} individuos | 🔧 Codificación: {encoding}",
                           font=('Arial', 9))
    status_label.pack(anchor=tk.W)
    
    # Initialize the first page
    update_page()
    
    root.mainloop()

def exportar_resultados_csv(best_individual, best_generation, final_avg_fitness, 
                            tipo_optimizacion, params, generations_data, encoding):
    """
    Exporta los resultados del algoritmo genético a un archivo CSV.
    
    Parámetros:
    - best_individual: El mejor individuo encontrado
    - best_generation: Generación en la que se encontró el mejor
    - final_avg_fitness: Fitness promedio de la población final
    - tipo_optimizacion: 'min' o 'max'
    - params: Diccionario con los parámetros del algoritmo
    - generations_data: Lista con las poblaciones de cada generación
    - encoding: Tipo de codificación ('binary' o 'real')
    """
    
    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultados_GA_{timestamp}.csv"
    
    # Calcular el valor real según tipo de optimización
    if tipo_optimizacion == "min":
        resultado = -best_individual.fitness
        promedio_fitness_real = -final_avg_fitness
    else:
        resultado = best_individual.fitness
        promedio_fitness_real = final_avg_fitness
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Encabezado
        writer.writerow(['=== RESULTADOS DEL ALGORITMO GENÉTICO ==='])
        writer.writerow([])
        
        # Parámetros de configuración
        writer.writerow(['PARÁMETROS DE CONFIGURACIÓN'])
        writer.writerow(['Parámetro', 'Valor'])
        writer.writerow(['Tamaño de población', params['POP_SIZE']])
        writer.writerow(['Número de alelos', params['DNA_LENGTH']])
        writer.writerow(['Número de generaciones', params['GENERATIONS']])
        writer.writerow(['Codificación', params['ENCODING']])
        writer.writerow(['Límites', f"{params['BOUNDS'][0]}, {params['BOUNDS'][1]}"])
        writer.writerow(['Método de selección', params['SELECTION_METHOD']])
        writer.writerow(['Método de cruce', params['CROSSOVER_METHOD']])
        writer.writerow(['Probabilidad de cruce (Pc)', params['PC']])
        writer.writerow(['Método de mutación', params['MUTATION_METHOD']])
        writer.writerow(['Probabilidad de mutación (Pm)', params['PM']])
        writer.writerow(['Tipo de optimización', tipo_optimizacion.upper()])
        writer.writerow([])
        
        # Resultados principales
        writer.writerow(['RESULTADOS PRINCIPALES'])
        writer.writerow(['Métrica', 'Valor'])
        writer.writerow([f'Valor {"mínimo" if tipo_optimizacion == "min" else "máximo"} encontrado', f'{resultado:.6f}'])
        writer.writerow(['Generación del mejor resultado', best_generation])
        writer.writerow(['Promedio del fitness (población final)', f'{promedio_fitness_real:.6f}'])
        writer.writerow(['Promedio del ADN del mejor individuo', f'{sum(best_individual.dna)/len(best_individual.dna):.6f}'])
        writer.writerow([])
        
        # ADN del mejor individuo
        writer.writerow(['ADN DEL MEJOR INDIVIDUO'])
        if encoding == 'real':
            writer.writerow(['Posición'] + [f'Alelo {i+1}' for i in range(len(best_individual.dna))])
            writer.writerow(['Valor'] + [f'{val:.6f}' for val in best_individual.dna])
        else:
            binary_dna = [1 if bit >= 0.5 else 0 for bit in best_individual.dna]
            writer.writerow(['Posición'] + [f'Bit {i+1}' for i in range(len(binary_dna))])
            writer.writerow(['Valor'] + binary_dna)
        writer.writerow([])
        
        # Evolución por generación
        writer.writerow(['EVOLUCIÓN POR GENERACIÓN'])
        writer.writerow(['Generación', 'Mejor Fitness', 'Fitness Promedio'])
        
        for gen_idx, pop in enumerate(generations_data[:-1]):  # Excluir la última que es duplicada
            best_fit = max([ind.fitness for ind in pop.individuals])
            avg_fit = sum([ind.fitness for ind in pop.individuals]) / len(pop.individuals)
            
            # Ajustar según tipo de optimización
            if tipo_optimizacion == "min":
                best_fit = -best_fit
                avg_fit = -avg_fit
            
            writer.writerow([gen_idx, f'{best_fit:.6f}', f'{avg_fit:.6f}'])
        
        writer.writerow([])
        writer.writerow(['POBLACIÓN FINAL (Todos los individuos)'])
        
        # Encabezados para la población final
        headers = ['#', 'Fitness']
        if encoding == 'real':
            headers += [f'Alelo_{i+1}' for i in range(params['DNA_LENGTH'])]
        else:
            headers += [f'Bit_{i+1}' for i in range(params['DNA_LENGTH'])]
        writer.writerow(headers)
        
        # Datos de todos los individuos de la población final
        final_pop = generations_data[-1]
        for i, ind in enumerate(final_pop.individuals):
            fitness_val = -ind.fitness if tipo_optimizacion == "min" else ind.fitness
            
            if encoding == 'real':
                row = [i+1, f'{fitness_val:.6f}'] + [f'{val:.6f}' for val in ind.dna]
            else:
                binary_dna = [1 if bit >= 0.5 else 0 for bit in ind.dna]
                row = [i+1, f'{fitness_val:.6f}'] + binary_dna
            
            writer.writerow(row)
    
    print(f"\n✅ Resultados exportados exitosamente a: {filename}")
    return filename

# MEJORA 3: Función de fitness con soporte para minimización/maximización
def get_user_function_tk():
    root = tk.Tk()
    root.withdraw()
    
    # Center the dialog
    root.update_idletasks()
    width = 400
    height = 150
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    
    func_str = simpledialog.askstring("Función de Fitness", 
                                    "Ingrese una función (ej: z = x + cos(y) o x + cos(y))\nVariables disponibles: x, y, z, w, ...",
                                    parent=root)
    root.destroy()
    
    if not func_str:
        print("No se ingresó ninguna función. Saliendo del programa.")
        sys.exit(1)
    
    python_expr = convert_to_python_syntax(func_str)
    
    # NUEVA FUNCIONALIDAD: Preguntar por tipo de optimización
    tipo_optimizacion = input("¿Desea encontrar el mínimo o el máximo de la función? (min/max): ").strip().lower()
    if tipo_optimizacion not in ["min", "max"]:
        print("Opción no válida. Se usará 'max' por defecto.")
        tipo_optimizacion = "max"
    
    def user_fitness(x):
        try:
            val = eval(python_expr, {"np": np, "x": x, "math": math})
            # Invertir el signo si buscamos el mínimo
            if tipo_optimizacion == "min":
                return -val
            else:
                return val
        except Exception as e:
            print(f"Error evaluando la función: {e}")
            sys.exit(1)
    
    return user_fitness, tipo_optimizacion

fitness_function, tipo_optimizacion = get_user_function_tk()

best_fitness = []
avg_fitness = []
best_generation = 0  # Rastrear en qué generación se encontró el mejor
best_value_so_far = float('-inf')  # Mejor valor encontrado hasta ahora
best_individual_ever = None  # Guardar el mejor individuo encontrado

pop = Population(POP_SIZE, DNA_LENGTH, ENCODING, BOUNDS)

generations_data = []
for gen in range(GENERATIONS):
    pop.evaluate(fitness_function)
    current_best = pop.get_best()
    current_best_fitness = current_best.fitness
    best_fitness.append(current_best_fitness)
    avg_fitness.append(pop.get_average_fitness())
    
    # Actualizar si encontramos un mejor valor
    if current_best_fitness > best_value_so_far:
        best_value_so_far = current_best_fitness
        best_generation = gen
        best_individual_ever = copy.deepcopy(current_best)
    
    # Store a copy of the population for this generation
    generations_data.append(copy.deepcopy(pop))

    # Selection (robust)
    try:
        selected = SELECTION_METHOD(pop, POP_SIZE)
        # Check for invalid output (empty or wrong size)
        if not selected or len(selected) != POP_SIZE:
            print(f"Advertencia: El método de selección '{params['SELECTION_METHOD']}' falló o devolvió tamaño incorrecto. Usando selección elitista.")
            selected = elitist_selection(pop, POP_SIZE)
    except Exception as e:
        print(f"Error en el método de selección '{params['SELECTION_METHOD']}': {e}. Usando selección elitista.")
        selected = elitist_selection(pop, POP_SIZE)

    # Crossover
    children = []
    for i in range(0, POP_SIZE, 2):
        p1 = selected[i]
        p2 = selected[(i+1)%POP_SIZE]
        c1_dna, c2_dna = CROSSOVER_METHOD(p1, p2, PC)
        children.append(type(p1)(c1_dna, ENCODING))
        children.append(type(p2)(c2_dna, ENCODING))
    children = children[:POP_SIZE]

    # Mutation
    for child in children:
        if MUTATION_METHOD == non_uniform_mutation:
            child.dna = MUTATION_METHOD(child.dna, PM, gen+1, GENERATIONS, ENCODING, BOUNDS)
        elif MUTATION_METHOD == swap_mutation:
            child.dna = MUTATION_METHOD(child.dna, PM)
        else:
            child.dna = MUTATION_METHOD(child.dna, PM, ENCODING, BOUNDS)

    pop.individuals = children

# Ensure all individuals have fitness before showing the final table
pop.evaluate(fitness_function)

# Verificar si la última generación tiene un mejor individuo
final_best = pop.get_best()
if final_best.fitness > best_value_so_far:
    best_value_so_far = final_best.fitness
    best_generation = GENERATIONS
    best_individual_ever = copy.deepcopy(final_best)

generations_data.append(copy.deepcopy(pop))

# MEJORA 4: Mostrar resultado final según tipo de optimización
final_avg_fitness = pop.get_average_fitness()

if tipo_optimizacion == "min":
    resultado = -best_individual_ever.fitness
    promedio_fitness_real = -final_avg_fitness
    print(f"\n🎯 Valor mínimo encontrado: {resultado:.6f}")
    print(f"📍 Ubicación: {best_individual_ever.dna}")
    print(f"📊 Promedio del ADN: {np.mean(best_individual_ever.dna):.6f}")
    print(f"📈 Promedio del fitness (población final): {promedio_fitness_real:.6f}")
    print(f"⏱️  Encontrado en la generación: {best_generation}")
else:
    resultado = best_individual_ever.fitness
    print(f"\n🎯 Valor máximo encontrado: {resultado:.6f}")
    print(f"📍 Ubicación: {best_individual_ever.dna}")
    print(f"📊 Promedio del ADN: {np.mean(best_individual_ever.dna):.6f}")
    print(f"📈 Promedio del fitness (población final): {final_avg_fitness:.6f}")
    print(f"⏱️  Encontrado en la generación: {best_generation}")

exportar_resultados_csv(
    best_individual=best_individual_ever,
    best_generation=best_generation,
    final_avg_fitness=final_avg_fitness,
    tipo_optimizacion=tipo_optimizacion,
    params=params,
    generations_data=generations_data,
    encoding=ENCODING
)

show_all_generations_window(generations_data, ENCODING)
plot_evolution(best_fitness, avg_fitness)