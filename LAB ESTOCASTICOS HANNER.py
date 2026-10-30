"""Laboratorio: procesos estocásticos y cadenas ocultas de Markov (HMM).
   Hanner Contreras 160005312
"""

import random


# Estados ocultos y observaciones posibles.
ESTADOS = ("Normal", "Bajo Ataque")
OBSERVACIONES = ("Trafico Bajo", "Trafico Alto")

# P(estado_t | estado_{t-1})
TRANSICION = {
	"Normal": {"Normal": 0.8, "Bajo Ataque": 0.2},
	"Bajo Ataque": {"Normal": 0.3, "Bajo Ataque": 0.7},
}
# P(observacion_t | estado_t)
EMISION = {
	"Normal": {"Trafico Bajo": 0.1, "Trafico Alto": 0.9},
	"Bajo Ataque": {"Trafico Bajo": 0.8, "Trafico Alto": 0.2},
}
INICIAL = {"Normal": 0.6, "Bajo Ataque": 0.4}


def muestrear(distribucion, rng):
	"""Extrae un valor de un diccionario {valor: probabilidad}."""
	r = rng.random()
	acumulada = 0.0
	for valor, probabilidad in distribucion.items():
		acumulada += probabilidad
		if r <= acumulada:
			return valor
	return next(reversed(distribucion))  # Protección ante redondeo.


def simular_hmm(n, semilla=7):
	"""Genera n estados ocultos y sus observaciones asociadas."""
	if n < 1:
		raise ValueError("n debe ser al menos 1")
	rng = random.Random(semilla)
	estado = muestrear(INICIAL, rng)
	estados, observaciones = [], []
	for t in range(n):
		if t:
			estado = muestrear(TRANSICION[estado], rng)
		estados.append(estado)
		observaciones.append(muestrear(EMISION[estado], rng))
	return estados, observaciones


def probabilidad_observaciones(observaciones):
	"""Algoritmo forward: calcula P(observaciones) sumando caminos ocultos."""
	if not observaciones:
		return 1.0
	alfa = {
		s: INICIAL[s] * EMISION[s][observaciones[0]] for s in ESTADOS
	}
	for observacion in observaciones[1:]:
		nuevo = {}
		for destino in ESTADOS:
			nuevo[destino] = EMISION[destino][observacion] * sum(
				alfa[origen] * TRANSICION[origen][destino]
				for origen in ESTADOS
			)
		alfa = nuevo
	return sum(alfa.values())


def camino_mas_probable(observaciones):
	"""Algoritmo de Viterbi: encuentra la secuencia oculta más probable."""
	if not observaciones:
		return []
	delta = {
		s: INICIAL[s] * EMISION[s][observaciones[0]] for s in ESTADOS
	}
	padres = []
	for observacion in observaciones[1:]:
		nuevo, padre_t = {}, {}
		for destino in ESTADOS:
			origen = max(
				ESTADOS,
				key=lambda s: delta[s] * TRANSICION[s][destino],
			)
			nuevo[destino] = (
				delta[origen]
				* TRANSICION[origen][destino]
				* EMISION[destino][observacion]
			)
			padre_t[destino] = origen
		delta = nuevo
		padres.append(padre_t)

	estado = max(ESTADOS, key=delta.get)
	camino = [estado]
	for padre_t in reversed(padres):
		estado = padre_t[estado]
		camino.append(estado)
	return list(reversed(camino))


if __name__ == "__main__":
	estados_reales, datos = simular_hmm(12)
	estimacion = camino_mas_probable(datos)
	print("Observaciones:             ", datos)
	print("Estados simulados (ocultos):", estados_reales)
	print("Estimacion de Viterbi:      ", estimacion)
	print(f"P(observaciones) = {probabilidad_observaciones(datos):.6f}")
