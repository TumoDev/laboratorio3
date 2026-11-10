"""Motor de recomendación que el estudiante debe completar.

Se proporcionan carga, validaciones y funciones auxiliares. Complete únicamente
los cuatro bloques señalados. Las puntuaciones de similitud no son probabilidades.
Este ejercicio no entrena un modelo supervisado: representa textos mediante
TF-IDF y actualiza un perfil con una regla explícita.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def cargar_catalogo(ruta: Path) -> pd.DataFrame:
    """Carga y valida el CSV proporcionado, sin modificar su orden."""
    recursos = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    columnas = ["id", "titulo", "descripcion", "etiquetas"]
    faltantes = set(columnas) - set(recursos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}.")
    if recursos.empty:
        raise ValueError("El catálogo no contiene recursos.")
    for columna in columnas:
        recursos[columna] = recursos[columna].str.strip()
        if (recursos[columna] == "").any():
            raise ValueError(f"La columna {columna} contiene valores vacíos.")
    if recursos["id"].duplicated().any():
        raise ValueError("Los identificadores de recursos deben ser únicos.")
    return recursos.reset_index(drop=True)


def cargar_perfiles(ruta: Path) -> dict:
    """Los intereses vacíos son válidos y se usan en la prueba T4."""
    with ruta.open(encoding="utf-8") as archivo:
        perfiles = json.load(archivo)
    if not isinstance(perfiles, dict) or not perfiles:
        raise ValueError("Se requiere un objeto JSON con perfiles de usuario.")
    for usuario_id, datos in perfiles.items():
        if not isinstance(usuario_id, str) or not usuario_id.strip():
            raise ValueError("Hay un identificador de usuario inválido.")
        if not isinstance(datos, dict):
            raise ValueError(f"El perfil {usuario_id} debe ser un objeto.")
        for campo in ("nombre", "intereses"):
            if not isinstance(datos.get(campo), str):
                raise ValueError(f"El campo {campo} de {usuario_id} debe ser texto.")
    return perfiles


def construir_indice(recursos: pd.DataFrame) -> tuple:
    """El vocabulario se aprende una vez y únicamente a partir de etiquetas."""
    vectorizador = TfidfVectorizer()
    matriz = vectorizador.fit_transform(recursos["etiquetas"]).toarray()
    return vectorizador, matriz


def crear_perfil(intereses: str, vectorizador: TfidfVectorizer) -> np.ndarray:
    """Devuelve un vector de una dimensión con el vocabulario del catálogo."""
    # TODO 1: transform (no fit) reutiliza el vocabulario e IDF del catálogo, así el
    # perfil queda en el mismo espacio que la matriz. Términos desconocidos -> ceros.
    return vectorizador.transform([intereses]).toarray()[0]


def calcular_puntuaciones(perfil: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """Devuelve una puntuación por fila del catálogo, en el mismo orden."""
    # TODO 2: cosine_similarity espera matrices; [perfil] es una matriz de 1 fila.
    # Con un perfil nulo, scikit-learn devuelve 0 para todos los recursos.
    return cosine_similarity([perfil], matriz)[0]


def recomendar(
    perfil: np.ndarray,
    recursos: pd.DataFrame,
    matriz: np.ndarray,
    vectorizador: TfidfVectorizer,
    top_k: int = 3,
) -> list[dict]:
    """Devuelve hasta top_k resultados positivos usando construir_resultado."""
    # TODO 3: se ordena por (-puntuación, id) para desempatar por id ascendente.
    # Solo se aceptan puntuaciones > 0; no se rellena con recursos sin coincidencia.
    puntuaciones = calcular_puntuaciones(perfil, matriz)
    positivos = [i for i, valor in enumerate(puntuaciones) if valor > 0]
    positivos.sort(key=lambda i: (-puntuaciones[i], recursos.iloc[i]["id"]))
    return [
        construir_resultado(i, puntuaciones[i], perfil, recursos, matriz, vectorizador)
        for i in positivos[:top_k]
    ]


def actualizar_perfil(perfil: np.ndarray, vector_recurso: np.ndarray) -> np.ndarray:
    """Actualiza el perfil cuando el usuario pulsa Me interesa sobre un recurso."""
    # TODO 4: la mezcla parte del vector actual (no de los intereses iniciales) y
    # normaliza a norma L2 = 1 para mantener una magnitud comparable entre
    # actualizaciones sucesivas. alpha = 0.6 es un parámetro didáctico.
    alpha = 0.6
    mezcla = (1 - alpha) * np.asarray(perfil, dtype=float) + alpha * np.asarray(
        vector_recurso, dtype=float
    )
    return normalize([mezcla])[0]


def obtener_indice(recursos: pd.DataFrame, recurso_id: str) -> int:
    """Valida el identificador antes de modificar un perfil."""
    coincidencias = recursos.index[recursos["id"] == recurso_id].tolist()
    if not coincidencias:
        raise ValueError(f"No existe el recurso {recurso_id}.")
    return int(coincidencias[0])


def explicar_coincidencias(
    perfil: np.ndarray,
    vector_recurso: np.ndarray,
    vectorizador: TfidfVectorizer,
) -> str:
    """La explicación se limita a términos compartidos con peso positivo."""
    vocabulario = vectorizador.get_feature_names_out()
    mascara = (np.asarray(perfil) > 0) & (np.asarray(vector_recurso) > 0)
    terminos = vocabulario[mascara].tolist()
    if not terminos:
        return "No hay términos compartidos con peso positivo."
    return "Términos compartidos: " + ", ".join(terminos) + "."


def construir_resultado(
    indice: int,
    puntuacion: float,
    perfil: np.ndarray,
    recursos: pd.DataFrame,
    matriz: np.ndarray,
    vectorizador: TfidfVectorizer,
) -> dict:
    """Función auxiliar para usar dentro de recomendar."""
    recurso = recursos.iloc[indice]
    return {
        "id": str(recurso["id"]),
        "titulo": str(recurso["titulo"]),
        "puntuacion": float(puntuacion),
        "explicacion": explicar_coincidencias(perfil, matriz[indice], vectorizador),
    }


def describir_perfil(perfil: np.ndarray, vectorizador: TfidfVectorizer) -> dict:
    """Representación legible del estado; no cambia el vector utilizado."""
    vocabulario = vectorizador.get_feature_names_out()
    pares = [(str(termino), float(peso)) for termino, peso in zip(vocabulario, perfil)
             if peso > 0]
    pares.sort(key=lambda par: (-par[1], par[0]))
    return {termino: round(peso, 6) for termino, peso in pares}
