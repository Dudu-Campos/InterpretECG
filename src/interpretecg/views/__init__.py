"""Visões derivadas do sinal: imagem e grafo, com seus readouts tabulares.

Os operadores são transformações determinísticas e sem treino de uma série 1D.
Procedência e divergências em relação à origem estão em `PROVENANCE.md`.
"""

from . import image_to_tabular, series_to_graph, series_to_image

__all__ = ["series_to_image", "series_to_graph", "image_to_tabular"]
