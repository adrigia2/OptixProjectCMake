# Build Instructions - Python Module

Guida per compilare e testare il modulo Python `depthMapModule`.

## Prerequisiti

### Software Richiesto
- ? Visual Studio 2019/2022 con C++17
- ? CUDA Toolkit 11.0+ (testato con 13.0)
- ? OptiX SDK 9.0.0
- ? CMake 3.10+
- ? Python 3.7+ con pip
- ? vcpkg (per OpenEXR e pybind11)

### Installazione Dipendenze

#### 1. Installa vcpkg (se non già installato)
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

#### 2. Installa le librerie C++ necessarie
```bash
# Dalla directory vcpkg
.\vcpkg install openexr:x64-windows
.\vcpkg install pybind11:x64-windows
.\vcpkg install glfw3:x64-windows
```

#### 3. Verifica Python
```bash
python --version  # Deve essere 3.7+
python -m pip install --upgrade pip
```

---

## Compilazione

### Opzione 1: Build con CMake GUI (Raccomandato per Windows)

1. **Apri CMake GUI**
   - Where is the source code: `C:/path/to/OptixProjectCMake`
   - Where to build the binaries: `C:/path/to/OptixProjectCMake/build`

2. **Configura**
   - Click su "Configure"
   - Seleziona "Visual Studio 17 2022" o "Ninja"
   - Specifica vcpkg toolchain file:
     ```
     C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
     ```

3. **Verifica Variabili Importanti**
   ```
   OptiX_INCLUDE       = C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include
   CUDA_TOOLKIT_ROOT   = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0
   PYTHON_EXECUTABLE   = C:/path/to/python.exe
   ```

4. **Generate e Build**
   - Click su "Generate"
   - Apri il progetto in Visual Studio
   - Build in **Release** mode (raccomandato per performance)

### Opzione 2: Build da Command Line

```bash
# Crea directory build
mkdir build
cd build

# Configura con vcpkg
cmake .. -G "Visual Studio 17 2022" ^
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCMAKE_BUILD_TYPE=Release

# Compila
cmake --build . --config Release

# Oppure con Ninja (più veloce)
cmake .. -G "Ninja" ^
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCMAKE_BUILD_TYPE=Release
ninja
```

---

## Output della Compilazione

Dopo la compilazione troverai:

### Windows
```
build/
??? depthMap/
    ??? depthMap.exe              # Eseguibile C++
    ??? depthMapModule.cp39-win_amd64.pyd  # Modulo Python
```

### Linux
```
build/
??? depthMap/
    ??? depthMap                  # Eseguibile C++
    ??? depthMapModule.cpython-39-x86_64-linux-gnu.so  # Modulo Python
```

---

## Test del Modulo Python

### 1. Test Importazione Basilare

```python
import sys
sys.path.append('C:/path/to/OptixProjectCMake/build/depthMap/Release')

try:
    import depthMapModule as optix
    print("? Modulo importato con successo!")
    print(f"? Versione: {optix.__version__ if hasattr(optix, '__version__') else 'N/A'}")
except ImportError as e:
    print(f"? Errore importazione: {e}")
```

### 2. Test Creazione Pipeline

```python
import depthMapModule as optix

try:
    pipeline = optix.OptiXPipeline()
    print("? Pipeline creata con successo!")
    
    # Verifica metodi disponibili
    methods = [m for m in dir(pipeline) if not m.startswith('_')]
    print(f"? Metodi disponibili: {methods}")
    
except Exception as e:
    print(f"? Errore creazione pipeline: {e}")
```

### 3. Test Completo con Dati di Test

```python
import depthMapModule as optix
import os

# Percorsi (modifica secondo il tuo setup)
MODEL = "C:/path/to/test_model.obj"
TRANSFORMS = "C:/path/to/transforms.json"
OUTPUT = "C:/temp/optix_test/"

# Crea directory output
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(OUTPUT + "ium/", exist_ok=True)
os.makedirs(OUTPUT + "depth/", exist_ok=True)

try:
    print("Inizializzazione pipeline...")
    pipeline = optix.OptiXPipeline()
    
    print(f"Caricamento modello: {MODEL}")
    pipeline.load_model(MODEL)
    
    if pipeline.is_model_loaded():
        print(f"? Modello caricato:")
        print(f"  - Vertici: {pipeline.get_vertex_count():,}")
        print(f"  - Triangoli: {pipeline.get_triangle_count():,}")
        
        print("\nGenerazione IUM...")
        pipeline.generate_ium(
            output_path=OUTPUT + "ium/",
            file_name="test_ium",
            image_type=optix.ImageResultType.OpenEXR,
            width=512,
            height=512
        )
        print("? IUM generato")
        
        if os.path.exists(TRANSFORMS):
            print("\nGenerazione depth maps...")
            pipeline.generate_depth_maps(
                transform_file=TRANSFORMS,
                output_dir=OUTPUT + "depth/",
                image_type=optix.ImageResultType.OpenEXR
            )
            print("? Depth maps generate")
        else:
            print(f"? Transforms file non trovato: {TRANSFORMS}")
        
        print("\n? Test completato con successo!")
    else:
        print("? Errore: modello non caricato")
        
except FileNotFoundError as e:
    print(f"? File non trovato: {e}")
except RuntimeError as e:
    print(f"? Errore runtime: {e}")
except Exception as e:
    print(f"? Errore generico: {e}")
    import traceback
    traceback.print_exc()
```

---

## Risoluzione Problemi

### Errore: `ImportError: DLL load failed`

**Causa**: Dipendenze DLL non trovate

**Soluzione**:
```bash
# Aggiungi vcpkg bin al PATH
set PATH=%PATH%;C:\path\to\vcpkg\installed\x64-windows\bin

# Oppure copia le DLL necessarie nella directory del modulo
copy C:\path\to\vcpkg\installed\x64-windows\bin\*.dll build\depthMap\Release\
```

### Errore: `ModuleNotFoundError: No module named 'depthMapModule'`

**Causa**: Modulo non nel PYTHONPATH

**Soluzione**:
```python
import sys
sys.path.insert(0, 'C:/path/to/build/depthMap/Release')
import depthMapModule
```

### Errore: `CUDA driver version is insufficient`

**Causa**: Driver NVIDIA obsoleto

**Soluzione**:
- Aggiorna i driver NVIDIA all'ultima versione
- Verifica con `nvidia-smi`

### Errore: `OptiX call failed`

**Causa**: OptiX non trovato o GPU non supportata

**Soluzione**:
1. Verifica installazione OptiX:
   ```
   C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\
   ```
2. Verifica GPU RTX (OptiX richiede GPU con RT cores)
3. Esegui in modalità debug per più dettagli

### Errore: Compilazione fallisce con linker errors

**Causa**: Librerie non trovate o versioni incompatibili

**Soluzione**:
```bash
# Reinstalla dipendenze vcpkg
.\vcpkg remove openexr:x64-windows
.\vcpkg remove pybind11:x64-windows
.\vcpkg install openexr:x64-windows
.\vcpkg install pybind11:x64-windows

# Pulisci e ricompila
cd build
cmake --build . --target clean
cmake --build . --config Release
```

### Warning: `PTX JIT compilation succeeded with warnings`

**Non critico**: PTX compilato con warning ma funzionante

**Opzionale**: Aggiungi flag di compilazione CUDA più stretti nel CMakeLists.txt

---

## Performance Tips

### 1. Build in Release Mode
```bash
cmake --build . --config Release
```
Release mode è **molto più veloce** di Debug (10-20x).

### 2. Ottimizzazioni CUDA
Nel `CMakeLists.txt`, aggiungi flag CUDA:
```cmake
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -use_fast_math 
    -O3
)
```

### 3. Test con Modelli Semplici
Per test veloci, usa modelli con:
- < 10,000 triangoli
- IUM resolution 512x512
- Poche depth maps (< 10 views)

---

## Deployment

### Creare un Package Distribuibile

1. **Copia file necessari**:
```bash
package/
??? depthMapModule.pyd
??? *.dll  (da vcpkg\installed\x64-windows\bin)
??? python_example.py
??? README.txt
```

2. **Crea installer (opzionale)**:
```python
# setup.py
from setuptools import setup

setup(
    name="depthMapModule",
    version="1.0.0",
    description="OptiX Depth Map Generator",
    py_modules=["depthMapModule"],
    package_data={'': ['*.pyd', '*.dll']},
)
```

3. **Build wheel**:
```bash
python setup.py bdist_wheel
pip install dist/depthMapModule-1.0.0-*.whl
```

---

## Verifica Installazione Completa

### Checklist
- [ ] CMake configura senza errori
- [ ] Compilazione completa senza errori
- [ ] File `.pyd` o `.so` generato
- [ ] Python importa il modulo
- [ ] `OptiXPipeline()` crea oggetto
- [ ] Test con modello reale funziona
- [ ] File output generati correttamente

### Script di Verifica Completo
```python
# verify_installation.py
import sys
import os

def verify():
    checks = []
    
    # Check 1: Import
    try:
        import depthMapModule as optix
        checks.append(("Import modulo", True, "OK"))
    except ImportError as e:
        checks.append(("Import modulo", False, str(e)))
        return checks
    
    # Check 2: Enum
    try:
        t = optix.ImageResultType.OpenEXR
        checks.append(("ImageResultType", True, "OK"))
    except Exception as e:
        checks.append(("ImageResultType", False, str(e)))
    
    # Check 3: Pipeline
    try:
        pipeline = optix.OptiXPipeline()
        checks.append(("OptiXPipeline", True, "OK"))
    except Exception as e:
        checks.append(("OptiXPipeline", False, str(e)))
    
    # Check 4: Metodi
    try:
        assert hasattr(pipeline, 'load_model')
        assert hasattr(pipeline, 'generate_ium')
        assert hasattr(pipeline, 'generate_depth_maps')
        checks.append(("Metodi pipeline", True, "OK"))
    except AssertionError:
        checks.append(("Metodi pipeline", False, "Metodi mancanti"))
    
    return checks

if __name__ == "__main__":
    print("Verifica installazione depthMapModule\n")
    print("="*50)
    
    results = verify()
    
    for name, success, msg in results:
        status = "?" if success else "?"
        print(f"{status} {name:.<30} {msg}")
    
    print("="*50)
    
    all_ok = all(success for _, success, _ in results)
    if all_ok:
        print("\n? Installazione verificata con successo!")
        sys.exit(0)
    else:
        print("\n? Alcuni controlli falliti")
        sys.exit(1)
```

---

## Link Utili

- [OptiX Documentation](https://raytracing-docs.nvidia.com/optix7/index.html)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [OpenEXR Technical Introduction](https://www.openexr.com/)
- [vcpkg Package Manager](https://vcpkg.io/)

---

## Supporto

Per problemi o domande:
1. Controlla questa guida
2. Verifica i log di compilazione
3. Testa con gli esempi forniti
4. Apri una issue su GitHub con:
   - Versione CMake, CUDA, OptiX
   - Log completo dell'errore
   - Sistema operativo e GPU

---

Ultimo aggiornamento: 2024
