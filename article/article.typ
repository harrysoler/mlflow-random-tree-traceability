#import "@preview/versatile-apa:7.2.0": (
  abstract-page, appendix, appendix-outline, title-page,
  versatile-apa as apa-style,
)
#import "@preview/orchid:0.1.0": (
  generate-link as orcid-link, logo-icon as orcid-logo,
)

#set text(lang: "es")

#set document(
  title: [Implementación de Línea Base Reproducible y Trazabilidad de Experimentos],
)

#show: apa-style.with(
  font-size: 12pt,
  running-head: "",
)

#title-page(
  authors: (
    (
      name: [Harrizon Soler],
      affiliations: (
        "ID-1",
      ),
    ),
  ),
  affiliations: (
    "ID-1": [Universidad Santo Tomás],
  ),

  course: [Seminario de Grado II],
  instructor: [Martha Susana Contreras Ortiz],
  due-date: datetime(day: 5, month: 11, year: 2026).display(),
)


El proyecto en general utiliza UV como gestor de venv y paquetes en local, asi mismo las ejecuciones de MLFlow fueron implementadas en marimo (una alternativa a Jupyter) y sus resultados fueron exportados y desplegados en #link("https://harrysoler-seminario-ii-activity-1.surge.sh"). El código fuente y el presente articulo se encuentran en #link("https://github.com/harrysoler/mlflow-random-tree-traceability").

#image("images/parallel_coordinates_plot.png")

#image("images/runs_list.png")

#image("images/comparative_table.png")
