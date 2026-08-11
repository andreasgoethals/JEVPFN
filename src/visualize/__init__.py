"""All plotting. `style` owns every colour and size; `figures` owns every saved file.

A notebook imports both, calls `style.apply()` once, and saves through
`figures.FigureSaver`. It never picks a colour and never calls `savefig` itself.
"""
