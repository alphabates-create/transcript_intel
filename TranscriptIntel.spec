# PyInstaller spec for TranscriptIntel.exe (one-folder build). Build with: .\build_exe.ps1
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

datas = [("app/app.py", "app")]
datas += collect_data_files("streamlit")            # front-end static files
datas += copy_metadata("streamlit")
datas += collect_data_files("plotly")
datas += collect_data_files("vaderSentiment")       # transcript scorer's word list
for pkg in ("anthropic", "pydantic", "pyarrow", "pandas", "scikit-learn"):
    datas += copy_metadata(pkg)

hiddenimports = (collect_submodules("streamlit") + collect_submodules("anthropic")
                 + ["src.enrich", "src.incidents", "src.insights", "src.llm", "src.load",
                    "src.pipeline", "src.screen", "src.sentiment", "src.topics", "sklearn.cluster", "sklearn.feature_extraction.text"])

# The local embedding model (torch) adds gigabytes; without it, clustering falls back to keyword vectors.
excludes = ["torch", "torchvision", "transformers", "sentence_transformers", "tensorflow", "umap", "numba",
            "llvmlite", "matplotlib", "IPython", "jupyter", "notebook", "ipykernel", "tkinter"]

a = Analysis(["run_app.py"], pathex=["."], datas=datas, hiddenimports=hiddenimports, excludes=excludes,
             hookspath=[], runtime_hooks=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TranscriptIntel", console=True,
          disable_windowed_traceback=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TranscriptIntel")
