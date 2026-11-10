"""Aplicación por consola proporcionada para el laboratorio evaluado.

La opción --comprobar funciona antes de completar el motor. --demo y el modo
interactivo requieren los cuatro bloques completados en motor.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import motor


BASE = Path(__file__).resolve().parent
SIN_COINCIDENCIAS = (
    "No hay coincidencias positivas. Declare intereses presentes en el catálogo."
)


def cargar_datos():
    recursos = motor.cargar_catalogo(BASE / "datos" / "recursos.csv")
    usuarios = motor.cargar_perfiles(BASE / "datos" / "perfiles.json")
    vectorizador, matriz = motor.construir_indice(recursos)
    return recursos, usuarios, vectorizador, matriz


def crear_estado(usuarios, vectorizador):
    return {
        usuario_id: motor.crear_perfil(datos["intereses"], vectorizador)
        for usuario_id, datos in usuarios.items()
    }


def crear_historial(usuarios):
    """Mantiene los eventos de cada usuario separados de su vector de intereses."""
    return {usuario_id: [] for usuario_id in usuarios}


def mostrar_resultados(resultados):
    if not resultados:
        print(SIN_COINCIDENCIAS)
        return
    for puesto, resultado in enumerate(resultados, start=1):
        print(f"{puesto}. {resultado['id']} | {resultado['titulo']} | "
              f"similitud={resultado['puntuacion']:.4f}")
        print(f"   {resultado['explicacion']}")


def mostrar_catalogo(recursos):
    for recurso in recursos.itertuples(index=False):
        print(f"{recurso.id} | {recurso.titulo} | {recurso.etiquetas}")


def seleccionar_usuario(usuarios):
    for usuario_id, datos in usuarios.items():
        print(f"{usuario_id}: {datos['nombre']}")
    while True:
        usuario_id = input("Usuario: ").strip().lower()
        if usuario_id in usuarios:
            return usuario_id
        print("Identificador de usuario no válido.")


def registrar_interes(estado, usuario_id, recurso_id, recursos, matriz, historial):
    if usuario_id not in estado or usuario_id not in historial:
        raise ValueError(f"No existe el usuario {usuario_id}.")
    indice = motor.obtener_indice(recursos, recurso_id)
    anterior = estado[usuario_id].copy()
    nuevo = motor.actualizar_perfil(anterior.copy(), matriz[indice].copy())
    nuevo = np.asarray(nuevo, dtype=float)
    if nuevo.shape != anterior.shape or not np.isfinite(nuevo).all():
        raise ValueError("El perfil actualizado debe tener la misma forma y valores finitos.")
    estado[usuario_id] = nuevo.copy()
    historial[usuario_id].append({
        "numero": len(historial[usuario_id]) + 1,
        "accion": "Me interesa",
        "recurso": recurso_id,
    })
    return anterior, nuevo.copy()


def comprobar(recursos, usuarios, vectorizador, matriz):
    print("Comprobación de preparación completada.")
    print(f"Recursos: {len(recursos)} | Usuarios: {len(usuarios)}")
    print(f"Matriz TF-IDF: {matriz.shape[0]} filas x {matriz.shape[1]} términos")
    print("Vocabulario: " + ", ".join(vectorizador.get_feature_names_out()))
    print("Se cargaron los datos y el índice sin ejecutar los bloques del estudiante.")


def interactivo(recursos, usuarios, vectorizador, matriz):
    estado = crear_estado(usuarios, vectorizador)
    historial = crear_historial(usuarios)
    usuario_id = seleccionar_usuario(usuarios)
    print("El estado se conserva durante esta ejecución. Reiniciar restaura el JSON inicial.")
    while True:
        print(f"\nUsuario activo: {usuario_id}")
        print("Perfil:", motor.describir_perfil(estado[usuario_id], vectorizador))
        print("Historial:", historial[usuario_id] or "Sin interacciones.")
        print("r: recomendar | c: catálogo | m: me interesa | u: cambiar usuario | s: salir")
        opcion = input("Opción: ").strip().lower()
        if opcion == "s":
            return
        if opcion == "r":
            mostrar_resultados(motor.recomendar(
                estado[usuario_id], recursos, matriz, vectorizador
            ))
        elif opcion == "c":
            mostrar_catalogo(recursos)
        elif opcion == "m":
            recurso_id = input("ID de cualquier recurso del catálogo: ").strip().upper()
            try:
                anterior, nuevo = registrar_interes(
                    estado, usuario_id, recurso_id, recursos, matriz, historial
                )
            except ValueError as error:
                print(error)
                continue
            print("Perfil anterior:", motor.describir_perfil(anterior, vectorizador))
            print("Perfil actualizado:", motor.describir_perfil(nuevo, vectorizador))
            mostrar_resultados(motor.recomendar(
                nuevo, recursos, matriz, vectorizador
            ))
        elif opcion == "u":
            usuario_id = seleccionar_usuario(usuarios)
        else:
            print("Opción no reconocida.")


def lista_recomendaciones_valida(resultados, recursos):
    """Comprueba el contrato de la lista sin construir un ranking alternativo."""
    if not isinstance(resultados, list) or not 1 <= len(resultados) <= 3:
        return False
    ids = [resultado["id"] for resultado in resultados]
    puntuaciones = [float(resultado["puntuacion"]) for resultado in resultados]
    claves = [(-puntuacion, recurso_id)
              for puntuacion, recurso_id in zip(puntuaciones, ids)]
    return bool(
        len(ids) == len(set(ids))
        and set(ids).issubset(set(recursos["id"]))
        and all(np.isfinite(puntuacion) and puntuacion > 0 for puntuacion in puntuaciones)
        and all(anterior <= siguiente for anterior, siguiente in zip(claves, claves[1:]))
    )


def ejecutar_demo(recursos, usuarios, vectorizador, matriz, carpeta_evidencias):
    """Guarda observaciones reales; la interpretación corresponde al estudiante."""
    estado = crear_estado(usuarios, vectorizador)
    historial = crear_historial(usuarios)
    esperados = {"u1", "u2", "u3", "u4"}
    if not esperados.issubset(estado):
        raise ValueError("La demostración requiere los perfiles u1, u2, u3 y u4.")
    indice_r04 = motor.obtener_indice(recursos, "R04")
    indice_r05 = motor.obtener_indice(recursos, "R05")
    resultados_u1 = motor.recomendar(estado["u1"], recursos, matriz, vectorizador)
    resultados_u2 = motor.recomendar(estado["u2"], recursos, matriz, vectorizador)
    perfil_u2_antes = estado["u2"].copy()
    historial_u2_antes = [evento.copy() for evento in historial["u2"]]
    scores_u2_antes = motor.calcular_puntuaciones(estado["u2"], matriz).copy()
    perfil_antes = estado["u1"].copy()
    scores_antes = motor.calcular_puntuaciones(perfil_antes, matriz).copy()
    pruebas = []

    condiciones_t1 = {
        "u1_lista_valida": lista_recomendaciones_valida(resultados_u1, recursos),
        "u2_lista_valida": lista_recomendaciones_valida(resultados_u2, recursos),
        "u1_primero_R02": bool(resultados_u1 and resultados_u1[0]["id"] == "R02"),
        "u2_primero_R07": bool(resultados_u2 and resultados_u2[0]["id"] == "R07"),
    }
    t1_ok = all(condiciones_t1.values())
    pruebas.append({"caso": "T1", "cumple": t1_ok,
                    "descripcion": "Dos perfiles iniciales sobre el mismo catálogo",
                    "condiciones": condiciones_t1,
                    "u1": resultados_u1, "u2": resultados_u2})

    historial_u1_antes = [evento.copy() for evento in historial["u1"]]
    registrar_interes(estado, "u1", "R04", recursos, matriz, historial)
    perfil_despues = estado["u1"].copy()
    scores_despues = motor.calcular_puntuaciones(perfil_despues, matriz).copy()
    resultados_despues = motor.recomendar(
        perfil_despues, recursos, matriz, vectorizador
    )
    condiciones_t2 = {
        "cambia_el_perfil": not np.allclose(perfil_antes, perfil_despues),
        "aumenta_puntuacion_R05": bool(scores_despues[indice_r05] > scores_antes[indice_r05]),
        "R04_pasa_al_primer_lugar": bool(
            resultados_despues and resultados_despues[0]["id"] == "R04"
        ),
        "se_conservan_los_12_candidatos": bool(
            len(recursos) == 12 and len(scores_antes) == 12 and len(scores_despues) == 12
        ),
        "se_registra_el_evento": historial["u1"] == [
            {"numero": 1, "accion": "Me interesa", "recurso": "R04"}
        ],
    }
    pruebas.append({"caso": "T2", "cumple": all(condiciones_t2.values()),
                    "descripcion": "u1 indica Me interesa sobre R04",
                    "condiciones": condiciones_t2,
                    "perfil_antes": motor.describir_perfil(perfil_antes, vectorizador),
                    "perfil_despues": motor.describir_perfil(perfil_despues, vectorizador),
                    "historial_antes": historial_u1_antes,
                    "historial_despues": [evento.copy() for evento in historial["u1"]],
                    "recomendaciones_antes": resultados_u1,
                    "recomendaciones_despues": resultados_despues})

    scores_u2_despues = motor.calcular_puntuaciones(estado["u2"], matriz)
    resultados_u2_despues = motor.recomendar(
        estado["u2"], recursos, matriz, vectorizador
    )
    condiciones_t3 = {
        "perfil_sin_cambios": bool(np.allclose(perfil_u2_antes, estado["u2"])),
        "puntuaciones_sin_cambios": bool(np.allclose(scores_u2_antes, scores_u2_despues)),
        "recomendaciones_sin_cambios": [r["id"] for r in resultados_u2] ==
                                      [r["id"] for r in resultados_u2_despues],
        "historial_sin_cambios": historial_u2_antes == historial["u2"],
    }
    t3_ok = all(condiciones_t3.values())
    pruebas.append({"caso": "T3", "cumple": t3_ok,
                    "descripcion": "La interacción de u1 no modifica a u2",
                    "condiciones": condiciones_t3,
                    "perfil_antes": motor.describir_perfil(perfil_u2_antes, vectorizador),
                    "perfil_despues": motor.describir_perfil(estado["u2"], vectorizador),
                    "historial_antes": historial_u2_antes,
                    "historial_despues": [evento.copy() for evento in historial["u2"]],
                    "recomendaciones_despues": resultados_u2_despues})

    t4_registros = []
    for usuario_id in ("u3", "u4"):
        resultados = motor.recomendar(
            estado[usuario_id], recursos, matriz, vectorizador
        )
        puntuaciones = motor.calcular_puntuaciones(estado[usuario_id], matriz)
        t4_registros.append({
            "usuario": usuario_id,
            "intereses": usuarios[usuario_id]["intereses"],
            "resultados": resultados,
            "puntuaciones_cero": bool(np.allclose(puntuaciones, 0)),
            "mensaje": SIN_COINCIDENCIAS if not resultados else "",
        })
    t4_ok = all(not registro["resultados"] and registro["puntuaciones_cero"]
                for registro in t4_registros)
    pruebas.append({"caso": "T4", "cumple": t4_ok,
                    "descripcion": "Perfil vacío y términos fuera del vocabulario",
                    "observaciones": t4_registros})

    carpeta_evidencias.mkdir(parents=True, exist_ok=True)
    comparacion = recursos[["id", "titulo", "etiquetas"]].copy()
    comparacion["puntuacion_antes"] = scores_antes
    comparacion["puntuacion_despues"] = scores_despues
    comparacion["cambio"] = scores_despues - scores_antes
    comparacion.to_csv(
        carpeta_evidencias / "puntuaciones_antes_despues.csv", index=False, encoding="utf-8"
    )
    reporte = {
        "nota": "Resultados de esta ejecución. La similitud no es una probabilidad.",
        "metodo": "TF-IDF de etiquetas y actualización normalizada de 0.4*p + 0.6*x",
        "candidatos": recursos["id"].tolist(),
        "retroalimentacion": {"usuario": "u1", "recurso": str(recursos.iloc[indice_r04]["id"]),
                              "accion": "Me interesa"},
        "historiales": historial,
        "pruebas": pruebas,
    }
    with (carpeta_evidencias / "reporte_demo.json").open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)
    for prueba in pruebas:
        print(f"{prueba['caso']}: {'Cumple' if prueba['cumple'] else 'No cumple'} | "
              f"{prueba['descripcion']}")
    print("\nu1 antes de valorar R04:")
    mostrar_resultados(resultados_u1)
    print("\nu1 después de valorar R04:")
    mostrar_resultados(resultados_despues)
    print("\nT4:")
    for registro in t4_registros:
        print(f"{registro['usuario']}: {registro['mensaje'] or 'Se encontraron resultados.'}")
    print(f"\nEvidencias guardadas en: {carpeta_evidencias}")
    print("Revise los archivos e interprete los resultados en README.md.")
    return all(prueba["cumple"] for prueba in pruebas)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Laboratorio de recomendación adaptativa")
    modos = parser.add_mutually_exclusive_group()
    modos.add_argument("--comprobar", action="store_true", help="Verifica datos y dependencias")
    modos.add_argument("--demo", action="store_true", help="Ejecuta T1-T4 y guarda evidencias")
    parser.add_argument("--evidencias", type=Path, default=BASE / "evidencias",
                        help="Directorio de salida de --demo")
    args = parser.parse_args(argv)
    try:
        recursos, usuarios, vectorizador, matriz = cargar_datos()
        if args.comprobar:
            comprobar(recursos, usuarios, vectorizador, matriz)
        elif args.demo:
            return 0 if ejecutar_demo(
                recursos, usuarios, vectorizador, matriz, args.evidencias
            ) else 1
        else:
            interactivo(recursos, usuarios, vectorizador, matriz)
    except NotImplementedError as error:
        print(error)
        print("La preparación puede comprobarse con: python app.py --comprobar")
        return 2
    except (ValueError, FileNotFoundError, pd.errors.ParserError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nEjecución finalizada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
