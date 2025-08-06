# 📊 Histogramas Cáncer de Mama

## 📌 Descripción
Este proyecto genera tres histogramas a partir de una base de datos de cáncer de mama:

1. **Histograma general** – Distribución del radio del tumor para todos los casos.  
2. **Histograma benignos** – Distribución del radio para casos diagnosticados como benignos.  
3. **Histograma malignos** – Distribución del radio para casos diagnosticados como malignos.  

El objetivo es visualizar cómo se distribuye el tamaño del tumor en cada tipo de diagnóstico.

---

## 🗂 Archivos
- **`01-histogramas.py`** → Script principal que genera los histogramas.  
- **`BD_CancerM.xlsx`** → Base de datos original en formato Excel.  
- **`histograma_general.png`** → Gráfico general.  
- **`histograma_benigno.png`** → Gráfico para tumores benignos.  
- **`histograma_maligno.png`** → Gráfico para tumores malignos.  

---

## ⚙️ Requisitos
Este proyecto usa **Python 3** y un entorno virtual.  
Librerías necesarias:
```bash
pandas
matplotlib
openpyxl


## 🚀 Cómo ejecutar

1. **Activar el entorno virtual**  
   En la carpeta del proyecto, abre la terminal y escribe:
   ```bash
   source venv/bin/activate

    python 01-histogramas.py
