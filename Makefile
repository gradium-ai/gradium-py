doc:
	uv run --with sphinx --with sphinx-autodoc-typehints --with myst-parser --with furo --with aiohttp --with numpy sphinx-build -b html docs docs/_build/html

.PHONY: doc
