# -*- coding: utf-8 -*-
import sys
import os
import re

# autodoc_mock_imports = ['_tkinter']
# import matplotlib
# if os.name == 'posix':
# matplotlib.use('TkAgg')

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../idtrackerai"))
sys.path.insert(0, os.path.abspath("../../idtrackerai/animals_detection"))
sys.path.insert(0, os.path.abspath("../../idtrackerai/crossings_detection"))
sys.path.insert(0, os.path.abspath("../../idtrackerai/fragmentation"))
sys.path.insert(0, os.path.abspath("../../idtrackerai/tracker"))
sys.path.insert(0, os.path.abspath("../../idtrackerai/postprocessing"))

version = ""
with open("../../idtrackerai/__init__.py", "r") as fd:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
    ).group(1)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
project = u"idtrackerai"
copyright = u"2018, Champalimaud Center for the Unknown"
author = u"Francisco Romero Ferrero, Mattia G. Bergomi"
version = version
release = version
language = "en"
exclude_patterns = ["_build"]
pygments_style = "sphinx"
todo_include_todos = False
themedir = os.path.join(os.pardir, "scipy-sphinx-theme", "_theme")
html_theme = "scipy"
html_theme_path = [themedir]
html_theme_options = {
    "edit_link": False,
    "sidebar": "left",
    "scipy_org_logo": True,
    "navigation_links": False,
    "rootlinks": [
        ("https://gitlab.com/polavieja_lab/idtrackerai.git", "GitLab repo")
    ],
}
html_sidebars = {"**": ["globaltoc.html", "sourcelink.html", "searchbox.html"]}
html_title = "%s v%s Manual" % (project, version)
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "idtrackerai"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
latex_documents = [
    (
        master_doc,
        "idtrackerai.tex",
        u"idtrackerai Documentation",
        u" Francisco Romero-Ferrero, Mattia G. Bergomi",
        "manual",
    ),
]
man_pages = [
    (master_doc, "idtrackerai", u"idtrackerai Documentation", [author], 1)
]
texinfo_documents = [
    (
        master_doc,
        "idtrackerai",
        u"idtrackerai Documentation",
        author,
        "idtrackerai",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# google analytics
googleanalytics_id = "UA-114600635-1"
# autoclass_content = "both"

# def skip(app, what, name, obj, would_skip, options):
#     if name == "__init__":
#         return False
#     return would_skip

# def setup(app):
#     app.connect("autodoc-skip-member", skip)

# autodoc_default_options = {
#     'members': True,
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': False,
# }
