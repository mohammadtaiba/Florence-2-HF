# Florence-2 lokal testen

Dieses Projekt führt Microsoft Florence-2 lokal aus. Das Modell wird beim ersten Start automatisch von Hugging Face heruntergeladen.

Standardmodell:

```text
microsoft/Florence-2-large-ft
```

Das Projekt verarbeitet einzelne Bilder oder alle Bilder aus dem Ordner `images`.

## Inhaltsverzeichnis

* [Voraussetzungen](#voraussetzungen)
* [Projektstruktur](#projektstruktur)
* [Windows-Einrichtung](#windows-einrichtung)
* [Linux-Einrichtung](#linux-einrichtung)
* [Bilder hinzufügen](#bilder-hinzufügen)
* [Alle Bilder nach einem Objekt durchsuchen](#alle-bilder-nach-einem-objekt-durchsuchen)
* [Ausgabe](#ausgabe)
* [Einzelnes Bild verarbeiten](#einzelnes-bild-verarbeiten)
* [Allgemeine Objekterkennung](#allgemeine-objekterkennung)
* [Bildbeschreibung](#bildbeschreibung)
* [Phrase Grounding](#phrase-grounding)
* [Texterkennung mit Positionen](#texterkennung-mit-positionen)
* [Unterordner verarbeiten](#unterordner-verarbeiten)
* [Verfügbare Aufgaben anzeigen](#verfügbare-aufgaben-anzeigen)
* [CPU oder GPU prüfen](#cpu-oder-gpu-prüfen)
* [Anderes Modell verwenden](#anderes-modell-verwenden)
* [Häufige Fehler](#häufige-fehler)

  * [Python 3.11 fehlt](#python-311-fehlt)
  * [PowerShell erkennt `--image` oder `--task` nicht](#powershell-erkennt---image-oder---task-nicht)
  * [CUDA-Speicher reicht nicht aus](#cuda-speicher-reicht-nicht-aus)
* [Typischer Aufruf](#typischer-aufruf)
* [Zusammenfassung](#zusammenfassung)



## Voraussetzungen

* Windows 10/11 oder Linux
* Python 3.11
* Internetzugang beim ersten Start
* optional: NVIDIA-Grafikkarte mit CUDA-Unterstützung

## Projektstruktur

```text
Florence-2-HF/
├── images/
├── outputs/
├── src/
│   └── florence2_hf/
│       ├── cli.py
│       ├── constants.py
│       ├── errors.py
│       ├── runner.py
│       └── visualizer.py
├── requirements.txt
├── setup_windows.cmd
├── setup_linux.sh
└── README.md
```

## Windows-Einrichtung

Python 3.11 installieren, falls es noch nicht vorhanden ist:

```powershell
py install 3.11
```

Projekt einrichten:

```powershell
cd C:\Users\Taiba\Downloads\Florence-2-HF
.\setup_windows.cmd
```

Die virtuelle Umgebung muss nicht aktiviert werden.

## Linux-Einrichtung

```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

## Bilder hinzufügen

Lege alle Bilder in den Ordner `images`:

```text
images/
├── bild_01.jpg
├── bild_02.png
└── bild_03.webp
```

Unterstützte Formate:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

## Alle Bilder nach einem Objekt durchsuchen

Beispiel: Alle Bilder nach Autos durchsuchen.

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task open_vocabulary_detection `
    --text-input "car" `
    --output-dir .\outputs
```

Das Modell wird einmal geladen und verarbeitet anschließend alle Bilder.

## Ausgabe

Für jedes Bild werden folgende Dateien erstellt:

```text
outputs/
├── bild_01.json
├── bild_01_marked.jpg
├── bild_02.json
└── bild_02_marked.jpg
```

Die JSON-Datei enthält erkannte Objekte, Labels und Koordinaten.

Das markierte Bild zeigt die erkannten Objekte mit Bounding Boxes oder Polygonen.

Wenn kein Objekt erkannt wird, wird nur die JSON-Datei gespeichert.



## Einzelnes Bild verarbeiten

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --image .\images\bild_01.jpg `
    --task open_vocabulary_detection `
    --text-input "car" `
    --output-dir .\outputs
```

## Allgemeine Objekterkennung

Florence-2 erkennt selbstständig bekannte Objekte:

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task od `
    --output-dir .\outputs
```

## Bildbeschreibung

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task caption `
    --output-dir .\outputs
```

Bei Bildbeschreibungen wird normalerweise kein markiertes Bild erstellt.

## Phrase Grounding

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task phrase_grounding `
    --text-input "A red car next to a building." `
    --output-dir .\outputs
```

## Texterkennung mit Positionen

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task ocr_with_region `
    --output-dir .\outputs
```

## Unterordner verarbeiten

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --recursive `
    --task open_vocabulary_detection `
    --text-input "car" `
    --output-dir .\outputs
```

Verwende eindeutige Dateinamen, damit sich Ausgabedateien nicht überschreiben.

## Verfügbare Aufgaben anzeigen

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli --list-tasks
```

## CPU oder GPU prüfen

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Ausgabe:

* `True`: GPU wird verwendet.
* `False`: Verarbeitung läuft über die CPU.

CPU erzwingen:

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task open_vocabulary_detection `
    --text-input "car" `
    --output-dir .\outputs `
    --device cpu
```

## Anderes Modell verwenden

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task open_vocabulary_detection `
    --text-input "car" `
    --output-dir .\outputs `
    --model-id microsoft/Florence-2-base-ft
```

## Häufige Fehler

### Python 3.11 fehlt

```powershell
py install 3.11
```

### PowerShell erkennt `--image` oder `--task` nicht

Zwischen den Befehlszeilen dürfen keine Leerzeilen stehen. Der Backtick muss das letzte Zeichen der Zeile sein.

### CUDA-Speicher reicht nicht aus

Verwende ein kleineres Modell:

```text
microsoft/Florence-2-base-ft
```

Oder führe die Verarbeitung über die CPU aus:

```powershell
--device cpu
```

## Typischer Aufruf

```powershell
.\.venv\Scripts\python.exe -m florence2_hf.cli `
    --input-dir .\images `
    --task open_vocabulary_detection `
    --text-input "car" `
    --output-dir .\outputs
```

## Zusammenfassung

Das Projekt verarbeitet alle Bilder aus `images`, speichert die Erkennungen als JSON-Dateien und erstellt markierte Ausgabebilder.

Die Dauer jedes Bildes und die Gesamtdauer werden im Terminal angezeigt.