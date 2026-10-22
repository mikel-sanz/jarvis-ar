"""Script de auditoría de Capa 0 de NEXO (The Notary & Gatekeeper).

Propósito:
    Validar la integridad arquitectónica, reglas globales de seguridad,
    ausencia de credenciales expuestas y ejecución limpia de pruebas
    unitarias antes de certificar el proyecto como Ready-to-Ship.

Uso:
    $ python nexo_audit.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Asegurar codificación UTF-8 en terminales Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR: Path = Path(__file__).resolve().parent


def audit_structure() -> Tuple[bool, str]:
    """Verifica la existencia de los archivos estructurales mandatorios."""
    required_files: List[Path] = [
        ROOT_DIR / ".gitignore",
        ROOT_DIR / ".env.example",
        ROOT_DIR / "requirements.txt",
        ROOT_DIR / ".nexo_status.json",
        ROOT_DIR / ".nexo" / "env_state.json",
        ROOT_DIR / "src" / "__init__.py",
        ROOT_DIR / "src" / "config.py",
        ROOT_DIR / "src" / "camera.py",
        ROOT_DIR / "src" / "vision_ai.py",
        ROOT_DIR / "src" / "manual_engine.py",
        ROOT_DIR / "src" / "ar_engine.py",
        ROOT_DIR / "src" / "app.py",
    ]

    missing = [str(f.relative_to(ROOT_DIR)) for f in required_files if not f.exists()]
    if missing:
        return False, f"Archivos faltantes: {', '.join(missing)}"
    return True, "Estructura de archivos completa y verificada."


def audit_security_rules() -> Tuple[bool, str]:
    """Verifica reglas globales: no pickle, no yaml.load inseguro, no secrets expuestos."""
    src_dir: Path = ROOT_DIR / "src"
    py_files = list(src_dir.glob("*.py"))

    # Patrones prohibidos
    forbidden_patterns = [
        (re.compile(r"\bimport\s+pickle\b"), "Uso prohibido de pickle"),
        (re.compile(r"yaml\.load\("), "Uso prohibido de yaml.load() no seguro"),
        (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "Posible API Key de Google expuesta en código"),
    ]

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                for pattern, msg in forbidden_patterns:
                    if pattern.search(content):
                        return False, f"{msg} en {py_file.name}"
        except OSError as err:
            return False, f"No se pudo leer {py_file.name}: {err}"

    return True, "Seguridad estática (SAST) y reglas de exclusión: OK."


def run_unit_tests() -> Tuple[bool, str]:
    """Ejecuta la suite de pruebas unitarias con pytest."""
    python_exe: Path = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    executable: str = str(python_exe) if python_exe.exists() else sys.executable

    cmd = [executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return True, "Todas las pruebas unitarias pasaron exitosamente."
        return False, f"Fallo en pruebas unitarias:\n{proc.stdout}\n{proc.stderr}"
    except OSError as err:
        return False, f"No se pudo invocar pytest: {err}"


def main() -> None:
    """Función principal de certificación de Capa 0."""
    print("=" * 60)
    print("🛡️  NEXO CAPA 0: AUDITORÍA DE CALIDAD Y SEGURIDAD")
    print("=" * 60)

    checks = [
        ("Estructura de Proyecto", audit_structure),
        ("Blindaje SAST & Reglas Globales", audit_security_rules),
        ("Pruebas Unitarias (pytest)", run_unit_tests),
    ]

    all_passed = True

    for name, check_fn in checks:
        print(f"\n[Verificando] {name}...")
        passed, details = check_fn()
        status_icon = "✅ OK" if passed else "❌ FALLO"
        print(f"Resultado: {status_icon} — {details}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CERTIFICACIÓN NEXO: REPORTE OK — PROYECTO LISTO")
        print("=" * 60)
        sys.exit(0)
    else:
        print("⚠️ CERTIFICACIÓN NEXO: SE DETECTARON ANOMALÍAS")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
