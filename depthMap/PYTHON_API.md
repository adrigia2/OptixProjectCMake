# OptiX Pipeline - Python Bindings

Bindings Python per la generazione di Inverse UV Mapping (IUM) e Depth Maps usando OptiX 7.

## Installazione

### Prerequisiti
- Python 3.7+
- CUDA Toolkit
- OptiX SDK 9.0.0
- Compilatore C++ con supporto C++17
- CMake 3.10+

### Compilazione del modulo

```bash
# Dal repository root
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Il modulo depthMapModule verrà generato nella directory build/depthMap/
```

### Aggiunta al PYTHONPATH

```python
import sys
sys.path.append('path/to/build/depthMap')
import depthMapModule as optix
```

## Utilizzo Base

### Import del modulo

```python
import depthMapModule as optix
```

### Workflow Completo

```python
# Crea la pipeline
pipeline = optix.OptiXPipeline()

# Esegui tutto in un colpo solo
pipeline.process_all(
    model_path="model.obj",
    transform_file="transforms.json",
    ium_output_path="output/ium/",
    depth_output_dir="output/depth/",
    ium_file_name="ium_result",
    image_type=optix.ImageResultType.OpenEXR,
    ium_width=1024,
    ium_height=1024
)
```

### Workflow Step-by-Step

```python
pipeline = optix.OptiXPipeline()

# Step 1: Carica il modello
pipeline.load_model("model.obj")

# Step 2: Genera Inverse UV Mapping
pipeline.generate_ium(
    output_path="output/ium/",
    file_name="my_ium",
    image_type=optix.ImageResultType.OpenEXR,
    width=2048,
    height=2048
)

# Step 3: Genera Depth Maps
pipeline.generate_depth_maps(
    transform_file="transforms.json",
    output_dir="output/depth/",
    image_type=optix.ImageResultType.OpenEXR
)
```

## API Reference

### Classe `OptiXPipeline`

#### Metodi

##### `load_model(model_path: str) -> None`
Carica un modello 3D da file OBJ.

**Parametri:**
- `model_path`: Percorso al file .obj

**Esempio:**
```python
pipeline.load_model("path/to/model.obj")
```

---

##### `generate_ium(output_path: str, file_name: str, image_type: ImageResultType = OpenEXR, width: int = 512, height: int = 512) -> None`
Genera la texture Inverse UV Mapping.

**Parametri:**
- `output_path`: Directory di output
- `file_name`: Nome del file (senza estensione)
- `image_type`: Formato output (BMP o OpenEXR)
- `width`: Larghezza texture in pixel
- `height`: Altezza texture in pixel

**Esempio:**
```python
pipeline.generate_ium(
    output_path="output/",
    file_name="ium_texture",
    image_type=optix.ImageResultType.OpenEXR,
    width=2048,
    height=2048
)
```

---

##### `generate_depth_maps(transform_file: str, output_dir: str, image_type: ImageResultType = OpenEXR) -> None`
Genera depth maps da file transforms.json.

**Parametri:**
- `transform_file`: Percorso al file transforms.json
- `output_dir`: Directory di output per le depth maps
- `image_type`: Formato output (BMP o OpenEXR)

**Esempio:**
```python
pipeline.generate_depth_maps(
    transform_file="transforms.json",
    output_dir="output/depth/",
    image_type=optix.ImageResultType.OpenEXR
)
```

---

##### `process_all(...) -> None`
Esegue l'intero pipeline: carica modello, genera IUM e depth maps.

**Parametri:**
- `model_path`: Percorso al file .obj
- `transform_file`: Percorso al transforms.json
- `ium_output_path`: Directory output IUM
- `depth_output_dir`: Directory output depth maps
- `ium_file_name`: Nome file IUM (default: "ium_output")
- `image_type`: Formato output (default: OpenEXR)
- `ium_width`: Larghezza IUM (default: 512)
- `ium_height`: Altezza IUM (default: 512)

**Esempio:**
```python
pipeline.process_all(
    model_path="model.obj",
    transform_file="transforms.json",
    ium_output_path="output/ium/",
    depth_output_dir="output/depth/"
)
```

---

##### `get_vertex_count() -> int`
Restituisce il numero di vertici del modello caricato.

**Ritorna:** Numero di vertici (0 se nessun modello caricato)

---

##### `get_triangle_count() -> int`
Restituisce il numero di triangoli del modello caricato.

**Ritorna:** Numero di triangoli (0 se nessun modello caricato)

---

##### `is_model_loaded() -> bool`
Verifica se un modello è stato caricato.

**Ritorna:** True se un modello è caricato, False altrimenti

---

### Enum `ImageResultType`

Tipo di formato per il salvataggio delle immagini.

**Valori:**
- `ImageResultType.BMP`: Formato bitmap (8-bit per canale, normalizzato)
- `ImageResultType.OpenEXR`: Formato OpenEXR (float32, valori grezzi)

**Esempio:**
```python
# Usa OpenEXR (raccomandato per precisione)
type = optix.ImageResultType.OpenEXR

# Oppure BMP per visualizzazione rapida
type = optix.ImageResultType.BMP
```

---

### Funzioni Utility

##### `set_log_level(level: int) -> None`
Imposta il livello di verbosità del logging.

**Parametri:**
- `level`: 0 (Error), 1 (Warning), 2 (Info), 3 (Debug)

**Esempio:**
```python
optix.set_log_level(2)  # Info level
```

---

## Formati Output

### Inverse UV Mapping (IUM)

#### OpenEXR (raccomandato)
- **Formato**: `.exr`
- **Canali**: RGB + Alpha
  - R: Posizione X nello spazio 3D (float32)
  - G: Posizione Y nello spazio 3D (float32)
  - B: Posizione Z nello spazio 3D (float32)
  - A: Mask (1.0 = valido, 0.0 = non valido)
- **Precisione**: Float32, valori grezzi

#### BMP
- **Formato**: `.bmp`
- **Canali**: RGB (8-bit per canale)
- **Valori**: Normalizzati in [0, 255]
- **Uso**: Preview e debug

### Depth Maps

#### OpenEXR (raccomandato)
- **Formato**: `.exr`
- **Canale**: Z (float32)
- **Valori**: Depth grezzo in unità world-space
- **Dimensione**: ~4 byte per pixel

#### BMP
- **Formato**: `.bmp`
- **Formato**: Grayscale 8-bit
- **Valori**: Normalizzati in [0, 255]
- **Uso**: Preview e debug

---

## Esempi Pratici

### Elaborazione Batch

```python
import os
import glob

models = glob.glob("models/*.obj")

for model_path in models:
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    
    pipeline = optix.OptiXPipeline()
    pipeline.load_model(model_path)
    
    # IUM
    pipeline.generate_ium(
        output_path=f"output/{base_name}/",
        file_name=f"{base_name}_ium"
    )
    
    # Depth maps
    if os.path.exists(f"transforms/{base_name}.json"):
        pipeline.generate_depth_maps(
            transform_file=f"transforms/{base_name}.json",
            output_dir=f"output/{base_name}/depth/"
        )
```

### Gestione Errori

```python
try:
    pipeline = optix.OptiXPipeline()
    pipeline.load_model("model.obj")
    
    if pipeline.is_model_loaded():
        print(f"Modello caricato: {pipeline.get_vertex_count()} vertici")
        
        pipeline.generate_ium("output/", "ium")
        pipeline.generate_depth_maps("transforms.json", "output/depth/")
        
except RuntimeError as e:
    print(f"Errore runtime: {e}")
except Exception as e:
    print(f"Errore generico: {e}")
```

### Verifica del Modello

```python
pipeline = optix.OptiXPipeline()
pipeline.load_model("model.obj")

print(f"Modello caricato: {pipeline.is_model_loaded()}")
print(f"Vertici: {pipeline.get_vertex_count():,}")
print(f"Triangoli: {pipeline.get_triangle_count():,}")
```

---

## Risoluzione Problemi

### Errore: "Model not loaded"
Assicurati di chiamare `load_model()` prima di `generate_ium()` o `generate_depth_maps()`.

### Errore: CUDA out of memory
Riduci la risoluzione IUM o elabora meno frame alla volta.

### File OpenEXR non leggibili
Verifica che il software di lettura supporti canali custom (Z per depth) e float32.

### Performance lente
- Verifica che CUDA sia installato correttamente
- Controlla che la GPU supporti OptiX
- Riduci la risoluzione se necessario

---

## Note

- I file `.exr` mantengono la precisione float32 completa
- I file `.bmp` sono utili per preview ma perdono precisione
- La mask nell'IUM indica quali pixel UV corrispondono alla superficie 3D
- Le depth maps usano valori in world-space units

---

## Licenza

Vedi il file LICENSE nel repository principale.

## Supporto

Per bug report e feature request, apri una issue su GitHub.
