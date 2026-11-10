# Laboratorio 3: aplicación inteligente adaptativa

Ingeniería Web Avanzada · OII436-1 · Segundo semestre 2026

## Identificación

- Integrantes: Francisco Cortes
- Fecha: 2026-11-09

## Objetivo y alcance

Recomendador de recursos de aprendizaje basado en contenido. Representa recursos e
intereses con TF-IDF y los compara con la similitud del coseno. Cuando el usuario indica
**Me interesa** sobre un recurso, actualiza su perfil y vuelve a calcular las
recomendaciones en la misma ejecución. Se implementa por consola; no se incluye Angular.

Se completaron solo los cuatro bloques TODO de `motor.py`. Los datos y `app.py` son los
del material base, sin cambios.

## Entorno y ejecución

- **Versión de Python utilizada:** 3.15.2 (macOS, instalada con Homebrew).
- **Comando de activación utilizado:** `source .venv/bin/activate` (macOS).

```bash
# 1. Crear el entorno virtual con Python 3.10 o posterior (en Windows: py -m venv .venv)
python3 -m venv .venv          # en este equipo: python3.15 -m venv .venv

# 2. Activarlo
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
.venv\Scripts\activate.bat         # Windows CMD

# 3. Instalar dependencias y comprobar
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
python app.py --comprobar

# 4. Ejecutar
python app.py --demo     # pruebas T1-T4 y evidencias en evidencias/
python app.py            # modo interactivo
```

`requirements.txt` se generó con `python -m pip freeze` después de instalar numpy, pandas
y scikit-learn en Python 3.15.2 (numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1). `.gitignore` excluye `.venv/`, `__pycache__/`, `*.pyc` y `.DS_Store`.

Resultado de `--comprobar`: 12 recursos, 4 usuarios, matriz TF-IDF de 12 filas x 17
términos. Vocabulario: angular, api, componentes, consultas, datos, formularios, funciones,
graficos, http, interfaces, pandas, python, sql, tablas, validacion, variables, web.

## Catálogo y perfil inicial

- `datos/recursos.csv` tiene las columnas `id`, `titulo`, `descripcion` y `etiquetas`.
  **Solo `etiquetas` interviene en el cálculo**: es lo que se pasa a
  `TfidfVectorizer.fit_transform` en `construir_indice`. `titulo` y `descripcion` solo se
  muestran al usuario; `id` identifica el recurso y sirve para desempatar.
- **Valores vacíos:** `cargar_catalogo` lee todo como texto (`keep_default_na=False`),
  quita espacios y lanza `ValueError` si alguna columna queda vacía. La app muestra
  `Error: ...` y termina sin calcular nada.
- **IDs repetidos:** `cargar_catalogo` lanza `ValueError` si `id` tiene duplicados.
  Así cada recurso queda asociado a una sola fila de la matriz.
- `datos/perfiles.json` tiene u1 (`python datos pandas`), u2
  (`angular componentes interfaces`), u3 (intereses vacíos) y u4 (`zzxyz123`). Los
  intereses vacíos son válidos y producen un vector nulo.

## Implementación (`motor.py`)

| Función | Implementación |
|---|---|
| `crear_perfil` | `vectorizador.transform([intereses]).toarray()[0]`. Reutiliza el vocabulario y los IDF del catálogo, sin volver a ajustar. Devuelve un vector 1D de 17 posiciones. Los términos desconocidos se ignoran (vector nulo). |
| `calcular_puntuaciones` | `cosine_similarity([perfil], matriz)[0]`: una puntuación por fila del catálogo, en el mismo orden. Con un perfil nulo devuelve ceros. |
| `recomendar` | Calcula las puntuaciones, se queda solo con las `> 0`, ordena por `(-puntuación, id)` (descendente con desempate por id ascendente), toma hasta `top_k=3` y construye cada resultado con `construir_resultado`. |
| `actualizar_perfil` | `mezcla = 0.4 * perfil_actual + 0.6 * vector_recurso` y luego `normalize([mezcla])[0]` (norma L2 = 1). Parte del vector actual, no de los intereses iniciales. |

Cada usuario tiene su propio vector en el diccionario `estado` y su propio historial en
`historial` (`app.py`). `registrar_interes` valida el ID con `obtener_indice` antes de
modificar algo, así que un ID desconocido muestra un mensaje y no cambia el estado.

## Pruebas funcionales (`python app.py --demo`)

| Caso | Entrada o interacción | Resultado esperado | Resultado observado | Estado |
|---|---|---|---|---|
| T1 | Recomendaciones iniciales para u1 y u2 | Rankings coherentes con cada interés, hasta 3 positivos, orden descendente, desempate por ID | u1: R02 (1.0000), R12 (0.4965), R03 (0.4326). u2: R07 (1.0000), R08 (0.5677), R09 (0.3087). Las dos listas son válidas, con R02 y R07 en primer lugar | **Cumple** |
| T2 | u1 indica Me interesa sobre R04 y se vuelve a recomendar | Cambia el vector de u1, cambia la puntuación de al menos un recurso no valorado y se mantienen los 12 candidatos | El perfil incorpora `api` y `web`. Nuevo ranking: R04 (0.8752), R02 (0.6880), R09 (0.4958). R05 pasa de 0.0000 a 0.4378 y R09 de 0.0000 a 0.4958. Historial de u1: `[{numero: 1, accion: Me interesa, recurso: R04}]`. Se mantienen los 12 candidatos | **Cumple** |
| T3 | Consultar u2 después de la interacción de u1 | El vector, el historial y las puntuaciones de u2 no cambian | Perfil de u2 igual (`componentes 0.6575, interfaces 0.5647, angular 0.4988`), historial vacío, mismas puntuaciones y ranking (R07, R08, R09) | **Cumple** |
| T4 | Recomendar para u3 (vacío) y u4 (`zzxyz123`) | Lista vacía y mensaje informativo, sin errores | Ambos con 12 puntuaciones en 0, lista `[]` y el mensaje "No hay coincidencias positivas. Declare intereses presentes en el catálogo." | **Cumple** |

**Interpretación:**

- **T1:** u1 y u2 no comparten términos en sus intereses, así que reciben rankings
  disjuntos sobre el mismo catálogo. R02 y R07 obtienen 1.0 porque sus etiquetas son
  exactamente los intereses declarados.
- **T2:** el cambio viene del vector actualizado. Los candidatos siguen siendo los mismos
  12 y no se quitó ningún recurso.
- **T3:** cada usuario tiene su propio vector y su propio historial. Además,
  `registrar_interes` trabaja sobre copias, así que la interacción de u1 no afecta a u2.
- **T4:** `transform` ignora los términos que no están en el vocabulario. El vector queda
  nulo, el coseno da 0 y `recomendar` descarta los ceros en vez de rellenar posiciones.

**Desempates:** después de la interacción, R05 y R06 quedan empatados (0.4378, porque
ambos comparten `web api`) y también R10 y R11 (0.1026). Ordenar por `(-puntuación, id)`
pone R05 antes de R06 y R10 antes de R11. Ninguno de estos empates cae dentro del top 3
en esta ejecución.

## Comparación antes y después de la valoración (u1 / R04)

**Perfil de u1** (pesos no nulos, de `describir_perfil`):

| Término | Antes | Después |
|---|---|---|
| pandas | 0.753247 | 0.376477 |
| datos | 0.465091 | 0.232455 |
| python | 0.465091 | 0.637116 |
| api | — | 0.446269 |
| web | — | 0.446269 |

**Recomendaciones:** antes R02, R12, R03; después R04, R02, R09.

**Puntuaciones de los 12 candidatos** (`evidencias/puntuaciones_antes_despues.csv`,
redondeadas al mostrarlas):

| id | Etiquetas | Antes | Después | Diferencia |
|---|---|---|---|---|
| R01 | python variables funciones | 0.1861 | 0.2549 | +0.0688 |
| R02 | python datos pandas | 1.0000 | 0.6880 | −0.3120 |
| R03 | python datos graficos | 0.4326 | 0.4044 | −0.0282 |
| R04 | python web api *(valorado)* | 0.2510 | 0.8752 | +0.6241 |
| **R05** | **web api http** | **0.0000** | **0.4378** | **+0.4378** |
| R06 | web api validacion | 0.0000 | 0.4378 | +0.4378 |
| R07 | angular componentes interfaces | 0.0000 | 0.0000 | 0.0000 |
| R08 | angular formularios interfaces | 0.0000 | 0.0000 | 0.0000 |
| **R09** | **angular web api** | **0.0000** | **0.4958** | **+0.4958** |
| R10 | sql datos consultas | 0.2053 | 0.1026 | −0.1027 |
| R11 | sql datos tablas | 0.2053 | 0.1026 | −0.1027 |
| R12 | python sql datos | 0.4965 | 0.4642 | −0.0324 |

### Cambio de puntuación de un recurso no valorado: R05

R05 (`web api http`) **no recibió ninguna valoración** y aun así su puntuación sube de
**0.0000 a 0.4378**. Antes, el perfil de u1 solo tenía `python`, `datos` y `pandas`, sin
ningún término en común con R05, así que el coseno era 0. Al valorar R04
(`python web api`), `actualizar_perfil` mezcla el 60 % del vector de R04 con el 40 % del
perfil anterior. Con eso el perfil incorpora `api` (0.4463) y `web` (0.4463), que son
justo los términos que R05 comparte con R04. El nuevo coseno es positivo por esa
coincidencia. R09 (`angular web api`) sube por la misma razón, de 0 a 0.4958, y entra al
top 3 desplazando a R12 y R03.

El efecto contrario también se ve. Como el perfil se normaliza, el peso relativo de
`pandas` y `datos` se reduce a la mitad aproximadamente. Por eso bajan R02 (−0.3120),
R10 y R11 (−0.1027), que dependen de esos términos. R07 y R08 no cambian (siguen en 0)
porque no comparten ningún término ni con el perfil inicial ni con R04.

Esto es evidencia de adaptación porque el único cambio entre las dos consultas es el
vector de u1, modificado automáticamente por el evento Me interesa. El catálogo, el
vectorizador y los 12 candidatos son los mismos.

### Problemas detectados

- No se detectaron fallos: las cuatro pruebas cumplen y los resultados coinciden con lo
  indicado en el enunciado (u1 parte con R02, R12, R03 y después de valorar R04 obtiene
  R04, R02, R09).
- El equipo solo tenía Python 3.9.6 del sistema, inferior al 3.10 que recomienda el
  material. Se instaló Python 3.15.2 con Homebrew, se recreó `.venv` con esa versión y se
  repitió todo el proceso (`--comprobar`, `--demo`, `pip freeze`). Las evidencias de esta
  entrega son de la ejecución con 3.15.2. Una prueba previa con 3.9.6 dio exactamente las
  mismas puntuaciones, como era de esperar: el cálculo no depende de la versión.
- `requirements.txt` fija versiones para Python 3.15. Con una versión de Python mucho
  más antigua, algunas de esas versiones pueden no estar disponibles. En ese caso se
  puede instalar `numpy pandas scikit-learn` sin versión.

## Preguntas

**1. ¿Qué información describe el contenido de los recursos y qué información constituye el perfil del usuario?**

El contenido de cada recurso lo describe la columna `etiquetas` de `datos/recursos.csv`
(por ejemplo, R04: `python web api`). Es lo único que se vectoriza en `construir_indice`,
y produce una fila de la matriz TF-IDF de 12 × 17. `titulo` y `descripcion` son solo para
presentación. El perfil del usuario tiene tres partes. La primera son los intereses
iniciales, en el campo `intereses` de `datos/perfiles.json` (u1: `python datos pandas`).
La segunda es el vector actual, que `crear_perfil` construye en el mismo espacio de 17
términos y se guarda en `estado[usuario_id]`. La tercera es el historial de interacciones
en `historial[usuario_id]` (`app.py`). La actualización se hace sobre el vector actual.

**2. ¿Por qué `fit_transform` sobre el catálogo y `transform` sobre los intereses? ¿Qué problema causaría un vectorizador distinto por perfil?**

`fit_transform` aprende del catálogo el vocabulario (los 17 términos, cada uno en una
posición fija) y los pesos IDF, que dependen de en cuántos recursos aparece cada término.
`transform` proyecta los intereses en ese mismo espacio: mismas dimensiones, mismo orden
de términos y mismos IDF. Así el coseno compara coordenadas que significan lo mismo. Si se
ajustara un vectorizador por perfil, el vocabulario saldría solo de los intereses
(u1 tendría 3 dimensiones y la matriz 17). Las dimensiones no coincidirían, o coincidirían
con significados distintos, y `cosine_similarity` fallaría o daría resultados sin sentido.
Además, los IDF calculados sobre un solo documento no reflejan la rareza de cada término
en el catálogo.

**3. ¿Por qué el mismo catálogo produce recomendaciones distintas para u1 y u2? ¿0,80 significa 80 % de probabilidad?**

El catálogo y la matriz son los mismos, pero cada usuario tiene un vector de perfil
distinto. El coseno mide la coincidencia de cada recurso con ese vector. u1 tiene peso en
`python`, `datos` y `pandas`, por eso le aparecen R02, R12 y R03. u2 tiene peso en
`angular`, `componentes` e `interfaces`, por eso le aparecen R07, R08 y R09. No, 0,80 no
es un 80 % de probabilidad. El coseno solo mide qué tan alineados están dos vectores de
pesos TF-IDF, es decir, cuánto se parecen sus etiquetas. No se calibró con datos reales
de gustos, no suma 1 entre recursos y no entiende el significado de las palabras. Un
recurso con 0,80 se parece más al perfil que uno con 0,40, pero eso no dice qué tan
probable es que le guste al usuario.

**4. ¿En qué se diferencia modificar manualmente los intereses iniciales de la adaptación implementada?**

Editar los intereses en `perfiles.json` es un cambio manual y externo. Requiere reiniciar
la aplicación, reemplaza el punto de partida y no deja registro de por qué cambió el
perfil. La adaptación implementada es automática y ocurre en la misma ejecución. El
**evento** que la provoca es la acción **Me interesa** sobre un recurso (opción `m` del
menú, o `registrar_interes(estado, "u1", "R04", ...)` en `--demo`). El **estado
modificado** es solo el vector actual del usuario activo (`estado["u1"]`), que pasa a
`normalize(0.4·perfil + 0.6·R04)`, más un evento nuevo en `historial["u1"]`. Los
intereses del JSON y el estado de u2 no se tocan. El **efecto observable** es que al
recalcular, el perfil incorpora `api` y `web`, R05 sube de 0 a 0.4378, R09 de 0 a 0.4958
y el ranking pasa de R02, R12, R03 a R04, R02, R09.

**5. Relación con el ciclo MAPE-K**

| Etapa | Elemento del programa |
|---|---|
| **Monitorizar** | La consola captura la opción `m` y el ID del recurso. `registrar_interes` recibe el evento Me interesa y lo agrega a `historial[usuario_id]`. |
| **Analizar** | `obtener_indice` valida que el recurso exista (un ID desconocido se rechaza sin cambios) y obtiene su vector `matriz[indice]`, es decir, las características que interesaron al usuario. |
| **Planificar** | La regla de `actualizar_perfil`: combinar `0.4 * perfil + 0.6 * recurso` (alpha = 0.6) y normalizar. Define cómo debe cambiar el perfil. |
| **Ejecutar** | Asignar `estado[usuario_id] = nuevo` y volver a llamar a `calcular_puntuaciones` y `recomendar`, que producen el nuevo ranking. |
| **Conocimiento** | El vectorizador y la matriz TF-IDF del catálogo, el diccionario `estado` con el vector de cada usuario y `historial` con sus eventos. Todas las etapas los consultan o actualizan. |

**6. Una limitación y una mejora concreta**

*Limitación:* el perfil depende mucho de la última interacción y no tiene memoria a largo
plazo. Con alpha = 0.6, cada Me interesa reduce los intereses anteriores al 40 %. Después
de dos o tres valoraciones de otro tema, los intereses declarados casi desaparecen (por
ejemplo, `pandas` pasa de 0.75 a 0.38 con una sola interacción). Tampoco hay
retroalimentación negativa: el sistema solo puede acercar el perfil a los recursos, nunca
alejarlo.

*Mejora propuesta:* agregar una acción **No me interesa** y una regla tipo Rocchio que use
el historial completo:
`perfil = normalize(β·intereses_iniciales + γ·promedio(recursos_positivos) − δ·promedio(recursos_negativos))`,
con los negativos recortados a cero. *Dato adicional que exige:* guardar en el historial
el tipo de valoración (positiva o negativa) y conservar el vector de intereses iniciales
por separado. *Cambio del algoritmo:* el perfil se recalcula a partir de todo el historial
en vez de mezclarse solo con el último recurso, lo que da más estabilidad. Los pesos
β, γ y δ tendrían que ajustarse con datos reales de uso, no fijarse a mano.
